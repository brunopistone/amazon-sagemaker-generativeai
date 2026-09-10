"""RLAIF reward: Amazon Bedrock's Converse API scores each completion.

Selected as ``--reward_funcs "judge"``. Calls ``Converse`` on the ``bedrock-runtime``
endpoint with a JSON-schema output contract, so the verdict shape is enforced
server-side.

Why Converse on ``bedrock-runtime`` rather than the Anthropic Messages API on
``bedrock-mantle``:

* **Structured output actually works.** ``output_config.format`` - which is what the
  Anthropic SDK's ``messages.parse(output_format=...)`` sends under the hood - is
  rejected with a 400 on ``bedrock-mantle``. The documented way to get
  schema-constrained output is Converse or InvokeModel on ``bedrock-runtime``.
* **One code path for every model.** Converse is model-agnostic, so Claude, Nova,
  Llama, DeepSeek, Mistral, gpt-oss and the rest of the catalogue all take the same
  request shape. No per-family client.
* **No bearer token to manage.** boto3 signs with SigV4 from the job's own AWS
  credentials, so there is no token to mint, cache, rotate or invalidate - and
  nothing bound to a particular event loop.
* **Endpoint-only features.** Guardrails, intelligent prompt routing, cross-Region
  inference profiles and prompt caching are ``bedrock-runtime`` only. Prompt caching
  matters here: every judge call repeats the same rubric preamble.

For an OpenAI-compatible endpoint instead - Bedrock's ``/openai/v1`` paths, Mantle, a
local vLLM/SGLang server, or a third-party provider - use
``--reward_funcs "judge_http"``.

``boto3`` is synchronous, so each call is dispatched to a worker thread from a pool
sized to ``--judge_max_concurrency``; ``score_batch``'s semaphore is what actually
bounds in-flight requests per rank.
"""

import asyncio
import atexit
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._judge import (
    JSON_INSTRUCTION,
    JudgeConfig,
    Unscorable,
    anthropic_thinking_kwargs,
    is_anthropic_model,
    parse_verdict,
    run_preflight,
    score_batch,
)
from ._registry import register_reward_factory

logger = logging.getLogger(__name__)

# Must be an *inference profile* id, not a bare foundation-model id: the current
# Claude models are profile-only, and Bedrock rejects a bare id with "Invocation of
# model ID ... with on-demand throughput isn't supported. Retry your request with the
# ID or ARN of an inference profile". Prefix with `us.` / `eu.` / `apac.` for a
# geography, or `global.` to route anywhere.
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# The verdict contract. `additionalProperties: false` plus an explicit `required`
# list are what make a JSON schema enforceable rather than advisory.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Quality score on the rubric's scale.",
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence justifying the score.",
        },
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}

# Errors that will not fix themselves. Raised as Unscorable so they are not retried:
# three attempts at a malformed request only burns the batch deadline. The startup
# preflight turns every one of these into an actionable abort before training begins.
PERMANENT_ERROR_CODES = frozenset(
    {
        "ValidationException",
        "AccessDeniedException",
        "ResourceNotFoundException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "ExpiredTokenException",
    }
)

# stopReason values that mean "successful call, no usable verdict".
_BLOCKED_STOP_REASONS = frozenset(
    {"content_filtered", "guardrail_intervened", "refusal"}
)


@register_reward_factory("judge")
def make_judge_reward_func(script_args: Any = None) -> Callable:
    """Build the Bedrock Converse judge reward from the run's ``--judge_*`` arguments.

    The reward is an ``async def`` closure: GRPOTrainer detects coroutine reward
    functions and awaits them on its own event loop, so every completion in a batch
    is judged concurrently (bounded by ``--judge_max_concurrency`` per rank).
    """
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        from botocore.exceptions import ClientError
    except ImportError as e:  # pragma: no cover - boto3 ships with sagemaker
        raise ValueError(
            "The 'judge' reward needs boto3: pip install boto3. For an "
            "OpenAI-compatible endpoint instead, use --reward_funcs 'judge_http'."
        ) from e

    cfg = JudgeConfig.from_script_args(script_args, default_model_id=DEFAULT_MODEL_ID)

    # Retries are owned by score_batch, which backs off with jitter and respects the
    # batch deadline. Letting botocore retry as well would multiply the latency of a
    # throttled call and make that deadline unpredictable.
    client = boto3.client(
        "bedrock-runtime",
        region_name=cfg.region,
        config=BotoConfig(
            retries={"max_attempts": 1, "mode": "standard"},
            connect_timeout=min(30.0, cfg.timeout),
            read_timeout=cfg.timeout,
            max_pool_connections=max(cfg.max_concurrency, 10),
        ),
    )

    # Reasoning knobs are per-family and mutually rejecting, so they are gated on the
    # family and sent in the shape that family's API generation expects:
    #   Claude 4.6+     thinking={"type":"adaptive","display":"summarized"}
    #                   + output_config.effort
    #   Claude <=4.5    thinking={"type":"enabled","budget_tokens":N} (rejects adaptive)
    #   everything else reasoning_effort, which Converse forwards verbatim - the shape
    #                   gpt-oss, GLM, Qwen and Grok accept.
    # `display: summarized` matters for a judge: without it Claude can return its
    # reasoning encrypted and the visible text empty, making every sample unscorable.
    extra_fields: Dict[str, Any] = {}
    if cfg.effort and cfg.effort.strip().lower() not in ("none", "off", "default", ""):
        if is_anthropic_model(cfg.model_id):
            extra_fields = anthropic_thinking_kwargs(cfg.model_id, cfg.effort)
        else:
            extra_fields = {"reasoning_effort": cfg.effort}
        logger.info(
            f"--judge_effort {cfg.effort!r} -> additionalModelRequestFields="
            f"{extra_fields}. If the preflight reports an empty or unparseable "
            "verdict, set --judge_effort none."
        )

    logger.info(
        f"Built 'judge' reward (Converse on bedrock-runtime, model={cfg.model_id}, "
        f"region={cfg.region}, scale=0-{cfg.scale:g}, "
        f"reasoning={cfg.effort if extra_fields else 'off'}, "
        f"concurrency={cfg.max_concurrency}/rank)"
    )

    # A plain thread pool: unlike an async client it is not bound to an event loop, so
    # the startup preflight and TRL's loop can share it.
    executor = ThreadPoolExecutor(
        max_workers=max(cfg.max_concurrency, 1), thread_name_prefix="judge"
    )
    atexit.register(executor.shutdown, wait=False)

    # Whether this model accepts Converse's structured-output contract. Support is
    # per-model rather than per-endpoint - verified against Bedrock: the Claude *-5
    # generation (opus-5, sonnet-5) rejects `outputConfig` with "output_config.format:
    # Extra inputs are not permitted", while 4.5/4.6 (opus-4-5, sonnet-4-5, haiku-4-5)
    # accept it. Held in a one-element list so the worker threads share the flip.
    use_schema = [cfg.structured_output != "off"]

    def is_schema_rejection(e: Any) -> bool:
        """Does this error mean the model will not accept ``outputConfig``?

        The refusal comes back from the *model*, not from Bedrock's request
        validation, so there is no dedicated error code to match on - only the
        message text.
        """
        error = e.response.get("Error", {})
        if error.get("Code") != "ValidationException":
            return False
        message = str(error.get("Message", "")).lower()
        return "output_config" in message or "outputconfig" in message

    def build_request(rendered: str, with_schema: bool) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "modelId": cfg.model_id,
            "messages": [{"role": "user", "content": [{"text": rendered}]}],
            # Sent whether or not the schema is enforced: harmless when it is, and
            # the only thing making the reply parseable when it is not.
            "system": [{"text": JSON_INSTRUCTION}],
            "inferenceConfig": {"maxTokens": cfg.max_tokens},
        }
        if with_schema:
            request["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(VERDICT_SCHEMA),
                            "name": "judge_verdict",
                            "description": "Score and one-sentence justification.",
                        }
                    },
                }
            }
        if extra_fields:
            request["additionalModelRequestFields"] = extra_fields
        return request

    def reraise(e: Any) -> None:
        """Re-raise a ClientError as Unscorable when retrying cannot help."""
        error = e.response.get("Error", {})
        code = error.get("Code", "")
        if code in PERMANENT_ERROR_CODES:
            raise Unscorable(
                f"{code}: {error.get('Message', e)} (model={cfg.model_id!r}, "
                f"region={cfg.region!r})"
            ) from e
        raise e  # throttling, timeouts, 5xx: score_batch retries with backoff

    def score_once(rendered: str) -> Optional[Tuple[float, str]]:
        """One blocking Converse call plus verdict extraction, in a worker thread."""
        # Capture the flag we are about to act on. Every completion in the batch is
        # scored concurrently, so if the condition below re-read `use_schema[0]` the
        # sibling threads that also sent the schema would see it already flipped to
        # False by the first refusal, skip their own fallback, and be reported as
        # permanently unscorable - losing a completion per rank on the first step.
        sent_with_schema = use_schema[0]
        try:
            response = client.converse(**build_request(rendered, sent_with_schema))
        except ClientError as first:
            if (
                sent_with_schema
                and cfg.structured_output == "auto"
                and is_schema_rejection(first)
            ):
                logger.warning(
                    f"{cfg.model_id} does not support Converse structured output: "
                    f"{first.response.get('Error', {}).get('Message', first)}. "
                    "Falling back to a JSON instruction in the system prompt with "
                    "tolerant parsing for the rest of this run. Set "
                    "--judge_structured_output off to skip this probe, or on to make "
                    "it a hard error."
                )
                use_schema[0] = False
                try:
                    response = client.converse(**build_request(rendered, False))
                except ClientError as second:
                    reraise(second)
            else:
                reraise(first)

        stop_reason = response.get("stopReason")
        if stop_reason in _BLOCKED_STOP_REASONS:
            raise Unscorable(
                f"the judge returned no verdict (stopReason={stop_reason!r}); the "
                "completion was refused or filtered rather than scored"
            )

        blocks = (response.get("output") or {}).get("message", {}).get("content") or []
        text = "".join(
            block["text"]
            for block in blocks
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
        if not text.strip():
            if stop_reason == "max_tokens":
                raise Unscorable(
                    f"hit the maxTokens cap ({cfg.max_tokens}) before emitting a "
                    "verdict - raise --judge_max_tokens, or set --judge_effort none: "
                    "a reasoning judge needs headroom for its reasoning *and* the "
                    "score."
                )
            raise Unscorable(
                f"no text in the response (stopReason={stop_reason!r}); if this "
                "happens for every sample the model may be returning reasoning only "
                f"for --judge_effort {cfg.effort!r}"
            )

        # With the schema enforced this is a plain json.loads; parse_verdict is used
        # regardless because it tries strict JSON first and costs nothing, and it is
        # what carries the fallback path where nothing is enforced.
        verdict = parse_verdict(text)
        if verdict is None:
            raise Unscorable(
                f"could not read a score out of the verdict {text.strip()[:200]!r}"
            )
        return verdict

    async def call(rendered: str) -> Optional[Tuple[float, str]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, score_once, rendered)

    async def teardown() -> None:
        """Nothing to release.

        The boto3 client and the thread pool are ordinary objects rather than
        loop-bound async clients, so the preflight's throwaway event loop leaves
        nothing behind for TRL's loop to trip over.
        """
        return None

    if cfg.preflight:
        run_preflight(cfg, call, teardown, "judge")

    async def judge_reward_func(
        prompts: List,
        completions: List,
        answer: Optional[List[str]] = None,
        log_metric: Optional[Callable] = None,
        log_extra: Optional[Callable] = None,
        **kwargs,
    ) -> List[Optional[float]]:
        """Rewards completions with a Bedrock judge quality score in [0, 1]."""
        return await score_batch(
            cfg,
            call,
            prompts,
            completions,
            answer=answer,
            log_metric=log_metric,
            log_extra=log_extra,
            metric_prefix="judge",
        )

    return judge_reward_func
