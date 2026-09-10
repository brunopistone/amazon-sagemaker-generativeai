"""Shared machinery for LLM-as-judge (RLAIF) rewards.

The backend-specific modules (``judge_reward.py``, ``judge_http_reward.py``) only
have to know how to turn one rendered judging prompt into a ``(score, critique)``
pair. Everything else - config, prompt rendering, tolerant score parsing,
concurrency, retries, timeouts, failure isolation and metric logging - lives here
so the two backends cannot drift apart.

Three facts about ``GRPOTrainer`` shape the design (see ``trl/trainer/grpo_trainer.py``):

  1. A reward function declared with ``async def`` is detected via
     ``inspect.iscoroutinefunction`` and awaited on a persistent event loop running
     in a daemon thread. Judge calls for a batch can therefore be fanned out with
     ``asyncio.gather`` - but anything loop-bound (a ``Semaphore``, an async HTTP
     client's connection pool) must be created *inside* the coroutine, not at
     factory time, because the factory runs on the main thread.
  2. A reward of ``None`` for a sample becomes ``NaN`` and is dropped from that
     sample's weighted reward sum. A judge call that throttles, times out or comes
     back unparseable therefore degrades one sample instead of killing the run -
     so nothing in here is allowed to raise once training has started.
  3. Rewards are computed per rank on that rank's local slice and gathered
     afterwards, so ``judge_max_concurrency`` is a *per-process* limit: the load
     the judge actually sees is ``world_size * judge_max_concurrency``.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ._common import extract_completion_text, render_message_content

logger = logging.getLogger(__name__)

# A judge call returns the score on the rubric's own scale plus its critique, or
# None if the response could not be turned into a score.
JudgeCall = Callable[[str], Awaitable[Optional[Tuple[float, str]]]]

# How an HTTP judge authenticates.
AUTH_API_KEY = "api_key"  # static bearer token from an environment variable
AUTH_BEDROCK_MANTLE = "bedrock_mantle"  # bedrock-mantle.{region}.api.aws
AUTH_BEDROCK_RUNTIME = "bedrock_runtime"  # bedrock-runtime.{region}.amazonaws.com
AUTH_MODES = (AUTH_API_KEY, AUTH_BEDROCK_MANTLE, AUTH_BEDROCK_RUNTIME)
# Both Bedrock endpoints authenticate identically - a short-lived bearer token minted
# from the caller's AWS credentials - and differ only in host and path.
AUTH_BEDROCK_MODES = (AUTH_BEDROCK_MANTLE, AUTH_BEDROCK_RUNTIME)

# Re-mint the Bedrock token this often. The token itself is valid for up to 12
# hours, but it is a SigV4-presigned artifact: when it is signed with *temporary*
# credentials - an EC2 instance role, IRSA, or the role a SageMaker training job
# runs under - it stops working the moment those session credentials rotate,
# which is typically well inside 12 hours. Minting is a local presign (no network
# call beyond resolving credentials), so re-minting often is nearly free and
# removes the whole class of "hour 6 of the run, every reward went NaN" failures.
BEDROCK_TOKEN_TTL = 2700.0  # 45 minutes

# Kept out of the rubric so rubric files never need brace-escaping. Sent as a system
# message by both backends: it is what makes the verdict parseable when the model's
# JSON shape is not being enforced server-side.
JSON_INSTRUCTION = (
    'Reply with a single JSON object and nothing else: {"score": <number>, '
    '"reasoning": "<one sentence>"}'
)

# ---------------------------------------------------------------------------
# HTTP wire protocols. Bedrock's OpenAI-compatible surface is not one API - the
# route depends on the model family, and sending a model to the wrong route is a
# 400/404 rather than a graceful degradation. Verified live against
# bedrock-mantle.us-east-1.api.aws:
#
#   chat_completions  {root}/v1/chat/completions      gpt-oss, GLM, Qwen,
#                                                     DeepSeek, Kimi, Nemotron...
#   responses         {root}/openai/v1/responses      the GPT-5.x family, which
#                                                     answers "isn't supported on
#                                                     this route" on the chat path
#   anthropic         {root}/anthropic/v1/messages    Claude, which explicitly
#                                                     rejects the chat path with
#                                                     "does not support the
#                                                     '/v1/chat/completions' API"
PROTOCOL_CHAT = "chat_completions"
PROTOCOL_RESPONSES = "responses"
PROTOCOL_ANTHROPIC = "anthropic"
PROTOCOLS = (PROTOCOL_CHAT, PROTOCOL_RESPONSES, PROTOCOL_ANTHROPIC)

# REASONING effort -> thinking budget_tokens, for Claude models old enough to take a
# token budget instead of an effort enum (<=4.5). Newer ones use the `adaptive` form
# with `output_config.effort`, where this map only sizes the max_tokens headroom.
_EFFORT_TO_TOKENS = {
    "low": 1024,
    "medium": 8192,
    "high": 16384,
    "xhigh": 24576,
    "max": 32768,
}


def is_anthropic_model(name: str) -> bool:
    """True if the model id names a Claude / Anthropic model.

    Substring-based so it survives Bedrock inference-profile prefixes
    (``us.``/``eu.``/``apac.``/``global.``), the ``anthropic.`` provider prefix and
    the bare ``claude-`` id. This is the gate for Anthropic-only request fields:
    every other family (gpt-oss, GPT-5.x, GLM, Qwen, DeepSeek, Kimi, Grok, Nova,
    Nemotron, Mistral, Gemma...) must never be sent them.
    """
    lowered = (name or "").lower()
    return "claude" in lowered or "anthropic." in lowered


def _claude_version(name: str) -> Optional[Tuple[int, int]]:
    """Best-effort ``(major, minor)`` from a Claude model id, or ``None``.

    ``claude-opus-4-8`` -> (4, 8); ``claude-3-7-sonnet`` -> (3, 7). A VERSION probe
    only, never an is-Claude signal: ``None`` means "Claude of unknown version,
    assume current", so callers must have confirmed the family first.
    """
    match = re.search(r"(\d+)[.\-](\d+)", name or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def anthropic_thinking_kwargs(name: str, effort: Optional[str]) -> Dict[str, Any]:
    """Extended-thinking fields in the shape the model's own API generation expects.

    Assumes ``name`` is already known to be Claude. Opus 4.7+ takes
    ``thinking={"type":"adaptive","display":"summarized"}`` plus
    ``output_config.effort``; Opus 4.6 the same without ``display``; 4.5 and older
    take ``thinking={"type":"enabled","budget_tokens":N}`` and *reject* adaptive.
    An unrecognised version is assumed to be newer than 4.7.

    ``display: summarized`` is the important one for a judge: without it a
    reasoning model can return its thinking encrypted and the visible text empty,
    which makes every completion unscorable.
    """
    version = _claude_version(name)
    use_adaptive = version is None or version >= (4, 6)
    if not use_adaptive:
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": _EFFORT_TO_TOKENS.get(effort or "medium", 8192),
            }
        }

    thinking: Dict[str, Any] = {"type": "adaptive"}
    if version is None or version >= (4, 7):
        thinking["display"] = "summarized"
    out: Dict[str, Any] = {"thinking": thinking}
    if effort:
        out["output_config"] = {"effort": effort}
    return out


def resolve_protocol(model_id: str, configured: Optional[str] = None) -> str:
    """Which HTTP protocol to speak for ``model_id``.

    An explicit ``--judge_protocol`` always wins. Otherwise it is inferred from the
    model id, because the routes are not interchangeable and the failure is a hard
    4xx: Claude -> ``anthropic``, the GPT-5.x family -> ``responses``, everything
    else -> ``chat_completions``.
    """
    if configured:
        configured = configured.strip().lower()
        if configured not in PROTOCOLS:
            raise ValueError(
                f"Unknown judge_protocol {configured!r}. Expected one of "
                f"{list(PROTOCOLS)}."
            )
        return configured

    lowered = (model_id or "").lower()
    if is_anthropic_model(lowered):
        return PROTOCOL_ANTHROPIC
    # Verified by probing every model in Mantle's GET /v1/models on all three
    # routes. The split does not follow provider, and two vendors are split
    # internally - so this is a recorded lookup, not a rule that can be reasoned out:
    #   openai.gpt-oss-*  chat        openai.gpt-5.*    responses
    #   google.gemma-3-*  chat        google.gemma-4-*  responses
    # The routes are strictly partitioned, so a wrong guess is a hard 400;
    # `judge_http` recovers by probing the others (see PROTOCOL_MISMATCH_MARKERS).
    if "xai." in lowered or "grok" in lowered:
        return PROTOCOL_RESPONSES
    if re.search(r"google\.gemma-([4-9]|\d\d)", lowered):
        return PROTOCOL_RESPONSES
    if re.search(r"(^|[./])openai\.gpt-[5-9]", lowered) and "oss" not in lowered:
        return PROTOCOL_RESPONSES
    return PROTOCOL_CHAT


# A 400 carrying one of these means "right endpoint, wrong route for this model".
# Bedrock phrases it two ways depending on which route rejected the request.
PROTOCOL_MISMATCH_MARKERS = (
    "isn't supported on this route",
    "is not supported on this route",
    "does not support the",
)


def is_protocol_mismatch(body: str) -> bool:
    """Whether an error body says the model is served on a different route."""
    lowered = (body or "").lower()
    return any(marker in lowered for marker in PROTOCOL_MISMATCH_MARKERS)


def other_protocols(current: str) -> Tuple[str, ...]:
    """The remaining protocols to probe, most likely first."""
    order = {
        PROTOCOL_CHAT: (PROTOCOL_RESPONSES, PROTOCOL_ANTHROPIC),
        PROTOCOL_RESPONSES: (PROTOCOL_CHAT, PROTOCOL_ANTHROPIC),
        PROTOCOL_ANTHROPIC: (PROTOCOL_CHAT, PROTOCOL_RESPONSES),
    }
    return order.get(current, ())



class Unscorable(Exception):
    """Raised by a backend when a *successful* call carries no usable score.

    Transport failures (throttling, timeouts, dropped connections) are retried;
    these are not. A safety refusal or a reply with no parseable number is the same
    on every attempt - judges run without sampling - so retrying only spends money
    to get the same answer. The sample simply gets no score from this reward.
    """


DEFAULT_RUBRIC = """You are grading a single response produced by a language model.

Score the response from 0 to {scale} on how well it answers the request: correct, \
directly responsive, complete, and free of padding. Judge substance only - length, \
confidence and formatting flourishes earn nothing on their own.

Use the full continuous range and be willing to use decimals. Responses that differ \
even slightly in quality must not receive the same score: your scores are compared \
against each other within a group of candidate responses, and identical scores carry \
no training signal at all.

Return the score and a one-sentence justification.

<request>
{prompt}
</request>
{reference_block}
<response>
{completion}
</response>"""

REFERENCE_BLOCK = """
<reference_answer>
{answer}
</reference_answer>
A response need not match the reference wording, only its substance."""


@dataclass
class JudgeConfig:
    """Judge settings, read off the run's ``ScriptArguments``."""

    model_id: str
    rubric: str
    scale: float = 10.0
    max_concurrency: int = 8
    retries: int = 3
    timeout: float = 120.0
    batch_timeout: Optional[float] = 900.0
    max_tokens: int = 2048
    # "none" means send no reasoning fields at all. Reasoning parameters are
    # model-family-specific and mutually rejecting, and Claude with adaptive thinking
    # plus an effort level returns its reasoning encrypted and the visible text
    # empty - which makes every completion unscorable. Off by default; opt in.
    effort: str = "none"
    # Whether to ask Bedrock to enforce the verdict's JSON schema server-side
    # (Converse's ``outputConfig``). Support is per-model, not per-endpoint: the
    # Claude *-5 generation rejects it outright while 4.5/4.6 accept it, so "auto"
    # tries once and falls back to a prompt instruction plus tolerant parsing.
    #   auto (default) - try, and downgrade permanently if the model refuses
    #   on             - always send; a refusal fails the run
    #   off            - never send; rely on the instruction and parse_verdict
    structured_output: str = "auto"
    # HTTP wire protocol for the judge_http backend: chat_completions / responses /
    # anthropic. Empty means infer it from the model id (see resolve_protocol).
    protocol: str = ""
    region: str = "us-east-1"
    base_url: Optional[str] = None
    auth: str = AUTH_API_KEY
    api_key_env: str = "JUDGE_API_KEY"
    log_critiques: bool = True
    preflight: bool = True
    uses_reference: bool = False

    @classmethod
    def from_script_args(
        cls, script_args: Any = None, *, default_model_id: Optional[str] = None
    ) -> "JudgeConfig":
        """Build a config from ``--judge_*`` script arguments.

        Every field falls back to the dataclass default, so a judge reward can also
        be built with ``script_args=None`` from a test or another script.
        """

        def get(name: str, default: Any) -> Any:
            value = getattr(script_args, f"judge_{name}", None)
            return default if value is None else value

        model_id = get("model_id", default_model_id)
        if not model_id:
            raise ValueError(
                "No judge model configured: set --judge_model_id (this backend has "
                "no default model)."
            )

        rubric = _load_rubric(get("rubric_path", None))
        cfg = cls(
            model_id=model_id,
            rubric=rubric,
            scale=float(get("score_scale", 10.0)),
            max_concurrency=int(get("max_concurrency", 8)),
            retries=int(get("retries", 3)),
            timeout=float(get("timeout", 120.0)),
            batch_timeout=_optional_positive(get("batch_timeout", 900.0)),
            max_tokens=int(get("max_tokens", 2048)),
            effort=str(get("effort", "none")),
            structured_output=str(get("structured_output", "auto")).strip().lower(),
            protocol=str(get("protocol", "") or "").strip().lower(),
            region=str(get("region", "us-east-1")),
            base_url=get("base_url", None),
            auth=str(get("auth", AUTH_API_KEY)),
            api_key_env=str(get("api_key_env", "JUDGE_API_KEY")),
            log_critiques=bool(get("log_critiques", True)),
            preflight=bool(get("preflight", True)),
        )
        if cfg.auth not in AUTH_MODES:
            raise ValueError(
                f"Unknown judge_auth {cfg.auth!r}. Expected one of {list(AUTH_MODES)}."
            )
        if cfg.structured_output not in ("auto", "on", "off"):
            raise ValueError(
                f"Unknown judge_structured_output {cfg.structured_output!r}. "
                "Expected 'auto', 'on' or 'off'."
            )
        if cfg.scale <= 0:
            raise ValueError(f"judge_score_scale must be positive, got {cfg.scale}.")
        if cfg.max_concurrency < 1:
            raise ValueError(
                f"judge_max_concurrency must be at least 1, got {cfg.max_concurrency}."
            )
        if cfg.retries < 1:
            raise ValueError(f"judge_retries must be at least 1, got {cfg.retries}.")
        if cfg.batch_timeout is not None and cfg.batch_timeout < cfg.timeout:
            raise ValueError(
                f"judge_batch_timeout ({cfg.batch_timeout}) is below judge_timeout "
                f"({cfg.timeout}), so no single call could ever finish. Raise it, or "
                "set it to 0 to remove the batch deadline entirely."
            )

        # Render once with placeholder values so a malformed rubric fails at startup
        # rather than on the first training step, and record whether it wants the
        # dataset's reference answer.
        cfg.uses_reference = (
            "{answer}" in cfg.rubric or "{reference_block}" in cfg.rubric
        )
        render_judge_input(cfg, "probe prompt", "probe completion", "probe answer")
        return cfg


def _optional_positive(value: Any) -> Optional[float]:
    """Coerce a timeout-style option, treating 0 / negative / None as "no limit"."""
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _rubric_candidates(rubric_path: str) -> List[str]:
    """Where to look for ``rubric_path``, in priority order.

    The working directory on a training host is not where the code was unpacked, and
    the unpack location differs per launcher: SageMaker's ModelTrainer mounts
    ``source_dir`` as the ``code`` channel at ``/opt/ml/input/data/code``, the classic
    Estimator uses ``/opt/ml/code``, Ray ships it somewhere else again, and a local
    run uses wherever ``python`` was invoked from. Resolving against ``__file__`` as
    well as the cwd makes ``judge_rubric_path: rubric.txt`` correct in all of them.

    An absolute path is honoured first, but its basename is still tried next to the
    code as a fallback - a hard-coded ``/opt/ml/code/rubric.txt`` should not fail a
    12-hour job when the file is sitting right beside ``train.py``.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(package_dir)
    name = os.path.basename(rubric_path)
    candidates = [rubric_path]
    if not os.path.isabs(rubric_path):
        candidates += [
            os.path.join(package_dir, rubric_path),
            os.path.join(script_dir, rubric_path),
        ]
    candidates += [
        os.path.join(package_dir, name),
        os.path.join(script_dir, name),
    ]
    return list(dict.fromkeys(candidates))


def _load_rubric(rubric_path: Optional[str]) -> str:
    if not rubric_path:
        return DEFAULT_RUBRIC

    candidates = _rubric_candidates(rubric_path)
    resolved = next((p for p in candidates if os.path.isfile(p)), None)
    if resolved is None:
        raise ValueError(
            f"Could not read --judge_rubric_path {rubric_path!r}: no such file. "
            f"Looked in: {candidates}. Ship the rubric inside the training "
            f"`source_dir` and reference it by name (e.g. 'rubric.txt'), or leave "
            f"--judge_rubric_path unset to use the built-in default rubric."
        )
    if resolved != rubric_path:
        logger.warning(
            f"--judge_rubric_path {rubric_path!r} does not exist; using {resolved!r} "
            "instead (found next to the training code)."
        )

    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            rubric = handle.read().strip()
    except OSError as e:
        raise ValueError(f"Could not read --judge_rubric_path {resolved!r}: {e}") from e
    if not rubric:
        raise ValueError(f"--judge_rubric_path {resolved!r} is empty.")
    if "{completion}" not in rubric:
        raise ValueError(
            f"The rubric in {resolved!r} must contain a '{{completion}}' placeholder "
            "(and usually '{prompt}'), otherwise the judge never sees the response it "
            "is meant to score."
        )
    return rubric


def render_prompt_text(prompt: Any) -> str:
    """Flatten a GRPO prompt to plain text for the judge.

    Standard-format prompts are already strings; conversational prompts are a list
    of message dicts, which are joined as ``role: content`` so the judge sees the
    whole exchange rather than only the last turn.

    Each message is rendered by ``render_message_content``, so an assistant turn
    whose payload lives in ``tool_calls`` / ``reasoning_content`` rather than
    ``content`` still appears. Reading ``content`` alone used to blank every such
    turn, which on a tool-use trajectory is every assistant turn in the prompt.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                role = message.get("role", "user")
                parts.append(f"{role}: {render_message_content(message)}")
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    return str(prompt)


def render_conversation_text(prompt: Any) -> str:
    """Render a prompt as a labelled transcript for the ``{conversation}`` field.

    An optional alternative to ``{prompt}`` for multi-turn rubrics. Two things it
    adds over the flat ``role: content`` join:

      - the instruction block (``system`` / ``developer``) is separated from the
        dialogue, so the judge can tell the task's standing rules and tool
        definitions apart from what happened in this episode;
      - the turn being responded to is named explicitly, which matters once the
        prompt ends on a ``tool`` message: without it the judge has to infer
        whether it is grading a reply to the user or a reaction to tool output.

    Degrades to essentially ``render_prompt_text`` for a single-turn prompt, so one
    rubric can serve both shapes.
    """
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt)

    instructions, dialogue = [], []
    for message in prompt:
        if not isinstance(message, dict):
            dialogue.append(("user", str(message)))
            continue
        role = message.get("role", "user")
        text = render_message_content(message)
        if role in ("system", "developer"):
            instructions.append(text)
        else:
            dialogue.append((role, text))

    sections = []
    if instructions:
        sections.append("[instructions]\n" + "\n\n".join(instructions))
    for index, (role, text) in enumerate(dialogue, start=1):
        sections.append(f"[turn {index}: {role}]\n{text}")
    if dialogue:
        sections.append(
            f"The response being graded answers turn {len(dialogue)} "
            f"({dialogue[-1][0]}) above."
        )
    return "\n\n".join(sections)


def render_judge_input(
    cfg: JudgeConfig, prompt: Any, completion: Any, answer: Optional[str] = None
) -> str:
    """Fill the rubric in for one (prompt, completion) pair."""
    reference_block = ""
    if answer and cfg.uses_reference:
        reference_block = REFERENCE_BLOCK.format(answer=answer)
    fields = {
        "prompt": render_prompt_text(prompt),
        "conversation": render_conversation_text(prompt),
        "completion": extract_completion_text(completion),
        "answer": answer or "",
        "reference_block": reference_block,
        "scale": _format_scale(cfg.scale),
    }
    try:
        return cfg.rubric.format(**fields)
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"The judge rubric references an unknown placeholder {e}. Available "
            f"placeholders: {sorted(fields)}. Escape a literal brace by doubling it."
        ) from e


def _format_scale(scale: float) -> str:
    return str(int(scale)) if scale == int(scale) else str(scale)


_SCORE_KEYS = ("score", "rating", "grade", "value")
_CRITIQUE_KEYS = ("reasoning", "justification", "critique", "explanation")

# "score: 8", 'score" : 8.5', "Rating = 7" - the number that is explicitly labelled
# as the score, wherever it sits in the prose.
_LABELLED_SCORE = re.compile(
    r"""["']?\b(?:score|rating|grade)\b["']?\s*[:=]\s*["']?(-?\d+(?:\.\d+)?)""",
    re.IGNORECASE,
)
_BARE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _score_from_mapping(data: Any) -> Optional[Tuple[float, str]]:
    if not isinstance(data, dict):
        return None
    # Match keys case-insensitively: judges routinely reply {"Score": 8}.
    lowered = {k.lower(): v for k, v in data.items() if isinstance(k, str)}
    for key in _SCORE_KEYS:
        if key not in lowered:
            continue
        try:
            score = float(lowered[key])
        except (TypeError, ValueError):
            continue
        critique = next((lowered[c] for c in _CRITIQUE_KEYS if lowered.get(c)), "")
        return score, str(critique)
    return None


def parse_verdict(text: str) -> Optional[Tuple[float, str]]:
    """Pull ``(score, critique)`` out of a judge response, tolerantly.

    Backends with native structured output do not need this; it exists for
    OpenAI-compatible servers, where JSON mode is not universally supported and the
    model may wrap its JSON in prose or a fenced code block.

    Tried in order, most trustworthy first:

      1. the whole reply as JSON;
      2. any JSON object embedded in the reply, located with ``raw_decode`` so that
         nested objects (``{"score": 8, "meta": {...}}``) parse as one value - a
         regex for ``{...}`` either stops at the first inner ``}`` or swallows
         trailing prose;
      3. a number explicitly *labelled* as the score (``score: 8``);
      4. only then the first bare number in the text.

    Step 3 exists because step 4 is dangerous on its own: a judge that opens with
    "On the 0-10 scale ..." makes the first number ``0``, and a confidently wrong
    0.0 reward corrupts the advantage far worse than no score at all.
    """
    if not text or not text.strip():
        return None

    verdict = _score_from_mapping(_loads_or_none(text))
    if verdict is not None:
        return verdict

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        verdict = _score_from_mapping(data)
        if verdict is not None:
            return verdict

    labelled = _LABELLED_SCORE.search(text)
    if labelled:
        return float(labelled.group(1)), text.strip()[:500]

    bare = _BARE_NUMBER.search(text)
    if bare:
        return float(bare.group(0)), text.strip()[:500]
    return None


def _loads_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def normalize_score(score: float, cfg: JudgeConfig) -> float:
    """Map a rubric-scale score into [0, 1], clamping out-of-range judges."""
    return min(max(score / cfg.scale, 0.0), 1.0)


async def score_batch(
    cfg: JudgeConfig,
    call: JudgeCall,
    prompts: List[Any],
    completions: List[Any],
    answer: Optional[List[str]] = None,
    log_metric: Optional[Callable[[str, float], None]] = None,
    log_extra: Optional[Callable[[str, list], None]] = None,
    metric_prefix: str = "judge",
) -> List[Optional[float]]:
    """Score a whole batch concurrently, one judge call per completion.

    Returns one reward in [0, 1] per completion, or ``None`` for a completion whose
    judge call could not be scored - GRPOTrainer turns that into ``NaN`` and drops
    this reward for that sample only.
    """
    answers: List[Optional[str]] = (
        list(answer) if answer is not None else [None] * len(completions)
    )
    if len(answers) != len(completions):  # defensive: TRL repeats columns for us
        answers = (answers + [None] * len(completions))[: len(completions)]

    # Created here, not in the factory: this coroutine runs on TRL's event loop.
    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    critiques: List[str] = [""] * len(completions)
    started = time.monotonic()

    async def score_one(index: int) -> Optional[float]:
        try:
            rendered = render_judge_input(
                cfg, prompts[index], completions[index], answers[index]
            )
        except Exception as e:  # noqa: BLE001 - startup validates the rubric, but a
            # malformed sample must still not take the step down.
            logger.warning(f"{metric_prefix}: could not render sample {index}: {e}")
            return None
        for attempt in range(cfg.retries):
            try:
                async with semaphore:
                    verdict = await asyncio.wait_for(
                        call(rendered), timeout=cfg.timeout
                    )
            except asyncio.CancelledError:
                raise
            except Unscorable as e:
                logger.warning(
                    f"{metric_prefix}: unscorable response, not retried: {e}"
                )
                return None
            except Exception as e:  # noqa: BLE001 - one sample must not fail the step
                logger.warning(
                    f"{metric_prefix}: call failed on attempt {attempt + 1}/"
                    f"{cfg.retries}: {type(e).__name__}: {e}"
                )
                verdict = None
            if verdict is not None:
                score, critique = verdict
                critiques[index] = critique
                return normalize_score(score, cfg)
            if attempt + 1 < cfg.retries:
                # Exponential backoff with jitter: every rank retries at once
                # otherwise, which is what turns a throttle into a thundering herd.
                await asyncio.sleep((2**attempt) * (0.5 + random.random()))
        return None

    # A per-call timeout does not bound the batch: waiting for a semaphore slot sits
    # outside it, so with enough completions a step could stall for
    # ceil(n / max_concurrency) * retries * timeout. The deadline below is the only
    # thing that puts a ceiling on how long one training step blocks on the judge.
    tasks = [asyncio.ensure_future(score_one(i)) for i in range(len(completions))]
    scores: List[Optional[float]] = []
    timed_out = 0
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=cfg.batch_timeout)
        for task in pending:
            task.cancel()
        if pending:
            # Let the cancellations settle so no task is still touching `critiques`.
            await asyncio.gather(*pending, return_exceptions=True)
        # Iterate `tasks`, not `done`, so the output order matches the input order.
        for task in tasks:
            if task in done and not task.cancelled():
                try:
                    scores.append(task.result())
                except Exception as e:  # noqa: BLE001 - score_one already guards, but
                    # a bug in here must not take the training step down either.
                    logger.warning(f"{metric_prefix}: scoring task failed: {e!r}")
                    scores.append(None)
            else:
                scores.append(None)
                timed_out += 1
        if timed_out:
            logger.warning(
                f"{metric_prefix}: batch deadline of {cfg.batch_timeout}s hit with "
                f"{timed_out}/{len(tasks)} completions still unscored. Raise "
                "--judge_batch_timeout, raise --judge_max_concurrency, or lower "
                "--judge_retries / --judge_timeout."
            )

    failures = sum(1 for score in scores if score is None)
    if log_metric is not None:
        log_metric(f"{metric_prefix}/calls", float(len(scores)))
        log_metric(f"{metric_prefix}/failure_rate", failures / max(len(scores), 1))
        log_metric(f"{metric_prefix}/batch_seconds", time.monotonic() - started)
        # Same keys on every rank, every step: TRL gathers these across processes.
        log_metric(f"{metric_prefix}/timeout_rate", timed_out / max(len(scores), 1))
    if failures:
        logger.warning(
            f"{metric_prefix}: {failures}/{len(scores)} completions could not be "
            "scored; they fall back to the other reward functions for this step."
        )
    # One value per sample, always - a short column would misalign the table.
    if cfg.log_critiques and log_extra is not None:
        log_extra(f"{metric_prefix}_critique", critiques)
    return scores


# A deliberately trivial pair: any working judge should score it without fuss, so a
# failure here is about configuration (credentials, model id, URL, token budget)
# rather than about the rubric being hard to apply.
PREFLIGHT_PROMPT = "What is 2 + 2?"
PREFLIGHT_COMPLETION = "2 + 2 = 4."


def run_preflight(
    cfg: JudgeConfig,
    call: JudgeCall,
    teardown: Callable[[], Awaitable[None]],
    metric_prefix: str,
) -> None:
    """Make one real judge call at startup, and refuse to train if it fails.

    Everything this catches - a stale API key, a model id the endpoint does not
    serve, a base URL that 404s, a ``max_tokens`` too small for a reasoning judge -
    otherwise shows up as *every sample unscorable*, which does not crash the run.
    Training then burns GPU hours optimising against a reward that is NaN everywhere,
    and the only clue is a ``failure_rate`` of 1.0 buried in the metrics. Failing here
    instead costs one judge call.

    Runs on the main thread before ``GRPOTrainer`` exists, so there is no event loop
    yet and ``asyncio.run`` is safe. ``teardown`` releases whatever the probe bound to
    that throwaway loop - an HTTP client, the Mantle token lock - since reusing any of
    it on TRL's loop would fail.
    """
    rendered = render_judge_input(cfg, PREFLIGHT_PROMPT, PREFLIGHT_COMPLETION)

    async def probe() -> Optional[Tuple[float, str]]:
        try:
            return await asyncio.wait_for(call(rendered), timeout=cfg.timeout)
        finally:
            await teardown()

    hint = (
        f"Set --judge_preflight false to skip this check. Judge: "
        f"model={cfg.model_id!r}, auth={cfg.auth!r}"
        + (f", base_url={cfg.base_url!r}" if cfg.base_url else "")
        + "."
    )
    try:
        verdict = asyncio.run(probe())
    except Unscorable as e:
        raise ValueError(
            f"The {metric_prefix!r} judge is reachable but returned no usable score "
            f"during startup preflight: {e}. Every completion would score NaN. {hint}"
        ) from e
    except asyncio.TimeoutError as e:
        raise ValueError(
            f"The {metric_prefix!r} judge did not answer within --judge_timeout "
            f"({cfg.timeout}s) during startup preflight. {hint}"
        ) from e
    except Exception as e:  # noqa: BLE001 - re-raised as an actionable config error
        raise ValueError(
            f"The {metric_prefix!r} judge preflight call failed: "
            f"{type(e).__name__}: {e}. {hint}"
        ) from e

    if verdict is None:
        raise ValueError(
            f"The {metric_prefix!r} judge returned an unparseable verdict during "
            f"startup preflight. {hint}"
        )

    score, critique = verdict
    logger.info(
        f"{metric_prefix}: preflight OK - judge scored a trivially correct answer "
        f"{score:g}/{_format_scale(cfg.scale)} "
        f"(reward {normalize_score(score, cfg):.3f}); critique: {critique[:120]!r}"
    )
    if score <= 0:
        logger.warning(
            f"{metric_prefix}: the judge scored a trivially *correct* answer "
            f"{score:g}/{_format_scale(cfg.scale)}. Check that --judge_score_scale "
            "matches the range the rubric actually asks for."
        )


async def aclose_client(client: Any) -> None:
    """Close an async client, whichever spelling it uses.

    httpx exposes ``aclose()``; the anthropic SDK's async clients expose ``close()``
    as the coroutine. Both are best-effort - a client that cannot be closed must not
    turn into a training failure.
    """
    for name in ("aclose", "close"):
        closer = getattr(client, name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001 - best effort
            pass
        return


def close_client_at_exit(client: Any, loop: Any) -> None:
    """Close ``client`` on the loop that owns it, at interpreter shutdown.

    An async client belongs to whichever loop first used it - TRL's, living in a
    daemon thread - so it can only be closed from there. Registered with ``atexit``
    *after* TRL registers its own loop shutdown; because atexit runs LIFO, this fires
    while the loop is still alive.
    """
    try:
        if loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(aclose_client(client), loop).result(timeout=5)
    except Exception:  # noqa: BLE001 - the process is exiting anyway
        pass


def resolve_api_key(cfg: JudgeConfig) -> Optional[str]:
    """Read the judge API key from the environment variable named in the config."""
    return os.environ.get(cfg.api_key_env) or None


def bedrock_openai_base_url(auth: str, region: str, protocol: str = PROTOCOL_CHAT) -> str:
    """Base URL for a Bedrock endpoint, for the given wire protocol.

    Mantle serves all three protocols on different roots::

        chat_completions  https://bedrock-mantle.{region}.api.aws/v1
        responses         https://bedrock-mantle.{region}.api.aws/openai/v1
        anthropic         https://bedrock-mantle.{region}.api.aws/anthropic

    ``bedrock-runtime`` serves the OpenAI chat surface at ``/openai/v1``. It has no
    ``/openai/v1/models`` listing (that is Mantle-only), and Claude is not on its
    OpenAI surface at all - use the ``judge`` reward (Converse) or Mantle's
    ``anthropic`` protocol for Claude.

    ``bedrock-runtime`` is the endpoint AWS recommends for new work: Guardrails,
    intelligent prompt routing, cross-Region inference profiles and prompt caching
    live there. Mantle adds server-side/pre-configured tool use, asynchronous
    inference and Projects/Workspaces, and serves a broader open-weights catalogue.
    Per-token pricing is identical, so choose on capability.
    """
    if auth == AUTH_BEDROCK_RUNTIME:
        root = f"https://bedrock-runtime.{region}.amazonaws.com"
        if protocol == PROTOCOL_ANTHROPIC:
            raise ValueError(
                "Claude is not served on bedrock-runtime's OpenAI-compatible "
                "surface. Use --reward_funcs 'judge' (Converse on bedrock-runtime), "
                "or --judge_auth bedrock_mantle, whose 'anthropic' protocol serves "
                "the Messages API."
            )
        return f"{root}/openai/v1"

    root = f"https://bedrock-mantle.{region}.api.aws"
    if protocol == PROTOCOL_RESPONSES:
        return f"{root}/openai/v1"
    if protocol == PROTOCOL_ANTHROPIC:
        return f"{root}/anthropic"
    return f"{root}/v1"


class BedrockTokenProvider:
    """Supplies (and quietly rotates) an Amazon Bedrock bearer token.

    Both Bedrock endpoints take ordinary AWS credentials, exchanged for a short-lived
    bearer token by ``aws_bedrock_token_generator`` (the same "Bedrock API key" an
    OpenAI SDK would send). Because a training run outlives the token, it cannot be
    captured once at startup and baked into a client's default headers - it is
    fetched per request and re-minted on a timer (see ``BEDROCK_TOKEN_TTL``) or on
    demand after the endpoint rejects it.
    """

    def __init__(self, region: str, ttl: float = BEDROCK_TOKEN_TTL) -> None:
        self._region = region
        self._ttl = ttl
        self._token: Optional[str] = None
        self._minted_at = 0.0
        self._lock: Optional[asyncio.Lock] = None
        self._refreshes = 0

    def _mint(self) -> str:
        try:
            from aws_bedrock_token_generator import provide_token
        except ImportError as e:
            raise ValueError(
                "judge_auth='bedrock_mantle'/'bedrock_runtime' needs the AWS token "
                "generator: pip install aws-bedrock-token-generator."
            ) from e
        return provide_token(region=self._region)

    def mint_now(self) -> str:
        """Mint synchronously, at startup, so absent credentials fail fast.

        Called from the factory on the main thread: an unusable credential chain
        should stop the job before the model loads, not surface as an unexplained
        wall of NaN rewards on the first training step.
        """
        self._token = self._mint()
        self._minted_at = time.monotonic()
        return self._token

    def invalidate(self) -> None:
        """Drop the cached token so the next request mints a fresh one."""
        self._token = None

    def detach(self) -> None:
        """Forget the event loop this provider latched onto.

        ``_lock`` binds to whichever loop first awaits it. The startup preflight runs
        on a throwaway ``asyncio.run`` loop, so without this the lock would still
        belong to that dead loop when training starts on TRL's loop, and the first
        token refresh would fail. The cached token itself is just a string and stays
        valid across loops.
        """
        self._lock = None

    async def get(self) -> str:
        """Return a usable token, re-minting if it is stale or was invalidated."""
        if self._lock is None:
            # Bound to TRL's event loop on first use, like the batch semaphore.
            self._lock = asyncio.Lock()
        async with self._lock:
            if (
                self._token is not None
                and time.monotonic() - self._minted_at < self._ttl
            ):
                return self._token
            # Resolving the credential chain can touch IMDS or the container
            # credential endpoint, so keep it off the event loop.
            loop = asyncio.get_running_loop()
            self._token = await loop.run_in_executor(None, self._mint)
            self._minted_at = time.monotonic()
            self._refreshes += 1
            logger.info(
                f"Refreshed Bedrock Mantle token (refresh #{self._refreshes}, "
                f"region={self._region})"
            )
            return self._token
