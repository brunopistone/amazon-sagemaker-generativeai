"""RLAIF reward: an OpenAI-compatible endpoint scores each completion.

Selected as ``--reward_funcs "judge_http"``. Works against anything that serves
``POST {base_url}/chat/completions``, with three ways to authenticate
(``--judge_auth``):

``bedrock_runtime``
    Amazon Bedrock's OpenAI-compatible surface on the endpoint AWS recommends for new
    work: ``https://bedrock-runtime.{region}.amazonaws.com/openai/v1``. This is where
    Guardrails, intelligent prompt routing, cross-Region inference profiles and
    prompt caching are available - so a model id may be a plain one
    (``openai.gpt-oss-120b``) or a cross-Region profile (``us.``/``global.`` prefix).

``bedrock_mantle``
    Amazon Bedrock Mantle: ``https://bedrock-mantle.{region}.api.aws/v1``. Serves a
    broad catalogue of open and third-party models and adds server-side and
    pre-configured tool use, asynchronous inference and Projects/Workspaces. Per-token
    pricing is identical to ``bedrock_runtime``, so choose on capability.

    Claude models are *not* reachable through Mantle's chat-completions path. They
    appear in its ``/v1/models`` listing, but ``/v1/chat/completions`` rejects them -
    Mantle serves them on its Anthropic-protocol path instead. For a Claude judge use
    ``--reward_funcs "judge"``, which calls Converse on ``bedrock-runtime``.

``api_key`` (default)
    A static bearer token read from the environment variable named by
    ``--judge_api_key_env``, so the key never lands in the YAML config or the job
    definition. This is the self-hosted path: vLLM, SGLang, TGI's OpenAI router, a
    SageMaker endpoint behind a compatible shim, or a hosted provider. Requires
    ``--judge_base_url``.

Both Bedrock modes derive their URL from ``--judge_region`` unless ``--judge_base_url``
overrides it, and both authenticate the same way: ordinary AWS credentials exchanged
for a short-lived bearer token that this module keeps rotating for the life of the run
(see ``_judge.BedrockTokenProvider``). A static Bedrock API key in the environment is
used instead when present, which lets the job run where the AWS credential chain is
not reachable.

The request is a plain ``httpx`` POST rather than the ``openai`` client - the wire
format is identical and it avoids a second HTTP stack in the training image.

``response_format`` is deliberately *not* sent: JSON mode is not universally
supported and a server that does not know the field rejects the whole request. The
JSON instruction goes in a system message instead, and the reply is parsed
tolerantly (``_judge.parse_verdict``), falling back to the first number in the text.
"""

import asyncio
import atexit
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._judge import (
    AUTH_BEDROCK_MODES,
    JSON_INSTRUCTION,
    PROTOCOL_ANTHROPIC,
    PROTOCOL_CHAT,
    PROTOCOL_RESPONSES,
    BedrockTokenProvider,
    JudgeConfig,
    Unscorable,
    aclose_client,
    anthropic_thinking_kwargs,
    bedrock_openai_base_url,
    close_client_at_exit,
    parse_verdict,
    resolve_api_key,
    resolve_protocol,
    is_protocol_mismatch,
    other_protocols,
    run_preflight,
    score_batch,
)
from ._registry import register_reward_factory

logger = logging.getLogger(__name__)

# Required header on the Anthropic Messages API.
ANTHROPIC_VERSION = "2023-06-01"


def _build_payload(cfg: JudgeConfig, protocol: str, rendered: str) -> Dict[str, Any]:
    """Request body in the shape the protocol's route expects.

    The three routes disagree on every field name that matters: the token cap is
    ``max_tokens`` / ``max_output_tokens`` / ``max_tokens``, the system prompt is a
    system *message* / ``instructions`` / a top-level ``system``, and reasoning is
    ``reasoning_effort`` / ``reasoning.effort`` / ``thinking``.
    """
    reasoning = cfg.effort and cfg.effort.strip().lower() not in (
        "none", "off", "default", "",
    )

    if protocol == PROTOCOL_ANTHROPIC:
        # Anthropic Messages: system is top-level, not a message.
        payload: Dict[str, Any] = {
            "model": cfg.model_id,
            "max_tokens": cfg.max_tokens,
            "system": JSON_INSTRUCTION,
            "messages": [{"role": "user", "content": rendered}],
        }
        if reasoning:
            thinking = anthropic_thinking_kwargs(cfg.model_id, cfg.effort)
            payload.update(thinking)
            # Extended thinking spends the same budget as the answer, so the cap has
            # to cover both or the verdict is truncated away.
            budget = (thinking.get("thinking") or {}).get("budget_tokens")
            if budget:
                payload["max_tokens"] = max(cfg.max_tokens, budget + 1024)
        return payload

    if protocol == PROTOCOL_RESPONSES:
        payload = {
            "model": cfg.model_id,
            "max_output_tokens": cfg.max_tokens,
            "instructions": JSON_INSTRUCTION,
            "input": [{"role": "user", "content": rendered}],
        }
        if reasoning:
            payload["reasoning"] = {"effort": cfg.effort}
        return payload

    payload = {
        "model": cfg.model_id,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": JSON_INSTRUCTION},
            {"role": "user", "content": rendered},
        ],
    }
    if reasoning:
        payload["reasoning_effort"] = cfg.effort
    return payload


def _protocol_path(protocol: str) -> str:
    """Route to POST, appended to the protocol's base URL."""
    if protocol == PROTOCOL_ANTHROPIC:
        return "/v1/messages"
    if protocol == PROTOCOL_RESPONSES:
        return "/responses"
    return "/chat/completions"


def _flatten_blocks(blocks: Any) -> str:
    """Join the ``text`` of a content-block list, ignoring thinking/tool blocks."""
    if not isinstance(blocks, list):
        return blocks if isinstance(blocks, str) else ""
    return "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def _extract_anthropic_text(cfg: JudgeConfig, payload: Any) -> str:
    """Pull the verdict text out of an Anthropic Messages response.

    ``content`` is a block list that may lead with ``thinking``/``redacted_thinking``
    blocks, so the text blocks are joined and the rest ignored.
    """
    if not isinstance(payload, dict) or "content" not in payload:
        raise Unscorable(f"unexpected response shape: {payload!r:.200}")

    stop_reason = payload.get("stop_reason")
    if stop_reason == "refusal":
        raise Unscorable("the judge refused to grade this completion")

    text = _flatten_blocks(payload.get("content"))
    if text.strip():
        return text

    if stop_reason == "max_tokens":
        raise Unscorable(
            f"hit the max_tokens cap ({cfg.max_tokens}) before emitting a verdict - "
            "raise --judge_max_tokens, or set --judge_effort none: extended thinking "
            "spends the same budget as the answer."
        )
    raise Unscorable(
        f"no text content in reply (stop_reason={stop_reason!r}); if this happens for "
        "every sample the model may be returning encrypted reasoning only"
    )


def _extract_responses_text(cfg: JudgeConfig, payload: Any) -> str:
    """Pull the verdict text out of a /responses reply.

    ``output`` is a list of items - reasoning items first on a thinking model - each
    with its own ``content`` block list. ``output_text`` is used when the server
    provides that convenience field.
    """
    if not isinstance(payload, dict):
        raise Unscorable(f"unexpected response shape: {payload!r:.200}")

    text = payload.get("output_text")
    if isinstance(text, list):
        text = "".join(part for part in text if isinstance(part, str))
    if isinstance(text, str) and text.strip():
        return text

    collected = []
    for item in payload.get("output") or []:
        if isinstance(item, dict) and item.get("type") in (None, "message"):
            collected.append(_flatten_blocks(item.get("content")))
    text = "".join(collected)
    if text.strip():
        return text

    status = payload.get("status")
    if status == "incomplete":
        reason = (payload.get("incomplete_details") or {}).get("reason")
        raise Unscorable(
            f"response incomplete (reason={reason!r}) - raise --judge_max_tokens, or "
            "set --judge_effort none: reasoning spends the same budget as the answer."
        )
    raise Unscorable(f"no output text in reply (status={status!r})")


def _extract_reply_text(cfg: JudgeConfig, payload: Any) -> str:
    """Pull the assistant text out of a chat-completions response.

    Raises ``Unscorable`` - never retried - when the reply cannot contain a score,
    with a message that names the cause, because "no score in reply: None" sends
    people looking at the rubric when the real problem is the token budget.
    """
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as e:
        raise Unscorable(f"unexpected response shape: {payload!r:.200}") from e

    if message.get("refusal"):
        raise Unscorable(f"judge refused: {message['refusal']!r:.200}")

    text = message.get("content")
    if isinstance(text, list):
        # Some OpenAI-compatible servers return content *blocks* rather than a
        # string. Left as a list it would reach parse_verdict and die on .strip(),
        # burning every retry on an AttributeError that names nothing useful.
        text = _flatten_blocks(text)
    if isinstance(text, str) and text.strip():
        return text

    # A reasoning judge fills `message.reasoning` first and only then emits
    # `content`, so a "length" finish means the budget ran out mid-thought and
    # `content` came back null. That is a configuration problem, not a bad rubric.
    if finish_reason == "length":
        raise Unscorable(
            f"hit the max_tokens cap ({cfg.max_tokens}) before emitting a verdict - "
            "raise --judge_max_tokens, a reasoning judge needs headroom for its "
            "reasoning *and* the score."
        )
    # `message.reasoning` is deliberately not used as a fallback: it restates the
    # rubric's scale ("score 0-10...") and hedges out loud ("maybe 8, or 9"), so the
    # first number in it is frequently wrong - and a confidently wrong 0.0 reward
    # does more damage to training than no reward at all.
    raise Unscorable(f"no content in reply (finish_reason={finish_reason!r})")


def _extract_for_protocol(cfg: JudgeConfig, protocol: str, payload: Any) -> str:
    if protocol == PROTOCOL_ANTHROPIC:
        return _extract_anthropic_text(cfg, payload)
    if protocol == PROTOCOL_RESPONSES:
        return _extract_responses_text(cfg, payload)
    return _extract_reply_text(cfg, payload)


def _raise_for_status(response: Any, tokens: Optional[BedrockTokenProvider]) -> None:
    """Turn an error response into the right kind of failure.

    ``score_batch`` retries anything that is not ``Unscorable``, so the split
    matters: a 400 from a model that does not speak this protocol would otherwise be
    re-sent for every sample of every step, and the reason the server gave would
    never reach the log - ``HTTPStatusError`` only carries the status line.
    """
    status = response.status_code
    if status in (401, 403) and tokens is not None:
        # Credentials rotated under us, or the token aged out early. Drop it so the
        # retry mints a fresh one instead of replaying a dead token.
        logger.warning(f"Bedrock rejected the token ({status}); re-minting.")
        tokens.invalidate()
        response.raise_for_status()
    if status >= 500 or status in (408, 429):
        response.raise_for_status()  # transient: retried by score_batch
    # Any other 4xx is a permanent client error - wrong model id, a field this
    # server rejects, a request over its limits. Retrying replays the same mistake.
    raise Unscorable(f"HTTP {status}: {response.text!r:.300}")


@register_reward_factory("judge_http")
def make_judge_http_reward_func(script_args: Any = None) -> Callable:
    """Build the HTTP judge reward from the run's ``--judge_*`` arguments.

    Requires ``--judge_model_id``; this backend has no default model. With
    ``--judge_auth bedrock_runtime`` or ``bedrock_mantle`` the endpoint is derived
    from ``--judge_region``, otherwise ``--judge_base_url`` is required.
    """
    try:
        import httpx
    except ImportError as e:
        raise ValueError(
            "The 'judge_http' reward needs httpx: pip install httpx."
        ) from e

    cfg = JudgeConfig.from_script_args(script_args)
    protocol = resolve_protocol(cfg.model_id, cfg.protocol)
    tokens: Optional[BedrockTokenProvider] = None
    headers: Dict[str, str] = {}

    if cfg.auth in AUTH_BEDROCK_MODES:
        base_url = cfg.base_url or bedrock_openai_base_url(
            cfg.auth, cfg.region, protocol
        )
        # A Bedrock API key in the environment wins: it lets the job run where the
        # AWS credential chain is not available. Otherwise mint from the chain and
        # keep re-minting, since the run outlives any single token.
        static_key = resolve_api_key(cfg) or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if static_key:
            headers["Authorization"] = f"Bearer {static_key}"
            auth_note = (
                f"static Bedrock key from ${cfg.api_key_env}/$AWS_BEARER_TOKEN_BEDROCK"
            )
        else:
            tokens = BedrockTokenProvider(cfg.region)
            tokens.mint_now()  # fail fast here rather than on step 1
            auth_note = "minted from AWS credentials, auto-refreshed"
    else:
        if not cfg.base_url:
            raise ValueError(
                "The 'judge_http' reward needs --judge_base_url, e.g. "
                "'http://judge-host:8000/v1' - or set --judge_auth bedrock_runtime "
                "(or bedrock_mantle) to use Amazon Bedrock, whose URL is derived "
                "from --judge_region."
            )
        base_url = cfg.base_url
        api_key = resolve_api_key(cfg)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        auth_note = f"from ${cfg.api_key_env}" if api_key else "none"

    url = base_url.rstrip("/") + _protocol_path(protocol)
    logger.info(
        f"Built 'judge_http' reward (model={cfg.model_id}, protocol={protocol}"
        f"{'' if cfg.protocol else ' [inferred]'}, url={url}, "
        f"scale=0-{cfg.scale:g}, concurrency={cfg.max_concurrency}/rank, "
        f"auth={cfg.auth} [{auth_note}])"
    )

    # Bound to TRL's event loop on first use, then reused so connections are pooled
    # across steps rather than renegotiated per batch.
    client: List[Any] = []

    def get_client():
        if not client:
            created = httpx.AsyncClient(
                headers=headers,
                timeout=cfg.timeout,
                limits=httpx.Limits(max_connections=cfg.max_concurrency),
            )
            client.append(created)
            # The client belongs to whichever loop first used it (TRL's, in a daemon
            # thread), so it can only be closed from there. atexit runs LIFO and this
            # is registered after TRL registers its own loop shutdown, so this fires
            # first, while the loop is still alive.
            loop = asyncio.get_running_loop()
            atexit.register(close_client_at_exit, created, loop)
        return client[0]

    async def teardown() -> None:
        """Release everything the preflight bound to its throwaway event loop."""
        while client:
            await aclose_client(client.pop())
        if tokens is not None:
            tokens.detach()

    def parse_or_raise(text: str) -> Tuple[float, str]:
        verdict = parse_verdict(text)
        if verdict is None:
            # Deterministic at temperature 0, so this is final for this sample.
            raise Unscorable(f"no score in reply: {text!r:.200}")
        return verdict

    # The route in use, mutable so a protocol probe latches for the whole run.
    active_protocol = [protocol]

    def _url_for(proto: str) -> str:
        if cfg.base_url:
            # An explicit base URL pins the endpoint; only the route varies.
            return cfg.base_url.rstrip("/") + _protocol_path(proto)
        return (
            bedrock_openai_base_url(cfg.auth, cfg.region, proto).rstrip("/")
            + _protocol_path(proto)
        )

    async def post_once(rendered: str, proto: str) -> Any:
        """One POST on ``proto``'s route. Raises ValueError if it is unavailable."""
        # A rotating token cannot live in the client's default headers, so it is
        # attached per request.
        request_headers: Dict[str, str] = {}
        if tokens is not None:
            request_headers["Authorization"] = f"Bearer {await tokens.get()}"
        if proto == PROTOCOL_ANTHROPIC:
            # Required by the Messages API; without it the request is rejected.
            request_headers["anthropic-version"] = ANTHROPIC_VERSION

        payload = _build_payload(cfg, proto, rendered)
        # Deterministic scoring: a sampled judge adds variance to the advantage,
        # which is pure noise in the gradient. Only the chat-completions route
        # reliably accepts it - verified against Bedrock, the newer models on the
        # other two routes reject it outright:
        #   Claude opus-5 / opus-4-8  400 "`temperature` is deprecated for this model"
        #   GPT-5.x on /responses     400 "'temperature' is not supported with this model"
        # Those families are near-deterministic at their default anyway.
        if proto == PROTOCOL_CHAT and "thinking" not in payload:
            payload["temperature"] = 0.0

        return await get_client().post(
            _url_for(proto), headers=request_headers or None, json=payload
        )

    async def call(rendered: str) -> Optional[Tuple[float, str]]:
        # Which route to use. Held in a list so a successful protocol probe is
        # remembered by every concurrent caller; each call captures the value it
        # acted on, so siblings that raced with a switch still retry properly.
        current = active_protocol[0]
        response = await post_once(rendered, current)

        if response.status_code == 400 and is_protocol_mismatch(response.text):
            # The model is served, just on a different route. Probe the others once
            # and latch the winner - this is what makes a newly-added model work
            # without having to re-derive the routing table by hand.
            for candidate in other_protocols(current):
                try:
                    probe = await post_once(rendered, candidate)
                except ValueError:
                    continue  # e.g. anthropic asked for on bedrock-runtime
                if probe.status_code < 400:
                    logger.warning(
                        f"{cfg.model_id} is not served on the {current!r} route; "
                        f"switching to {candidate!r} for the rest of this run. Set "
                        f"--judge_protocol {candidate} to skip this probe."
                    )
                    active_protocol[0] = candidate
                    return parse_or_raise(
                        _extract_for_protocol(cfg, candidate, probe.json())
                    )
            # None of the routes took it: report the original, most specific error.
            _raise_for_status(response, tokens)

        if response.status_code >= 400:
            _raise_for_status(response, tokens)
        return parse_or_raise(
            _extract_for_protocol(cfg, current, response.json())
        )

    if cfg.preflight:
        run_preflight(cfg, call, teardown, "judge_http")

    async def judge_http_reward_func(
        prompts: List,
        completions: List,
        answer: Optional[List[str]] = None,
        log_metric: Optional[Callable] = None,
        log_extra: Optional[Callable] = None,
        **kwargs,
    ) -> List[Optional[float]]:
        """Rewards completions with an HTTP judge's quality score in [0, 1]."""
        return await score_batch(
            cfg,
            call,
            prompts,
            completions,
            answer=answer,
            log_metric=log_metric,
            log_extra=log_extra,
            metric_prefix="judge_http",
        )

    return judge_http_reward_func
