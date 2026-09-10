"""GRPO reward functions, one per module, discovered automatically.

Adding a reward is adding a file to this directory:

    # reward_function/my_reward.py
    from ._common import extract_completion_text
    from ._registry import register_reward

    @register_reward("my_reward")
    def my_reward_func(completions, **kwargs):
        return [1.0 if "yes" in extract_completion_text(c) else 0.0
                for c in completions]

It is then selectable as ``--reward_funcs "my_reward"`` with no change to
``train_grpo.py``. Modules whose name starts with ``_`` are treated as private
helpers and skipped.

``--reward_funcs`` accepts a comma-separated list of either kind of spec:

  * a name registered in this package                  e.g. ``format``
  * ``module.path:function_name`` for a reward defined outside it
                                                       e.g. ``my_rewards:accuracy``

Reward callables must take ``(completions, **kwargs)`` and return one float per
completion; GRPOTrainer forwards every other dataset column as a keyword
argument. TRL logs each reward as ``rewards/<function __name__>/mean``, so a
reward's ``__name__`` is part of its observable contract - renaming a function
renames its metric.

Two further contract details, both used by the judge rewards in this package:

  * A reward declared ``async def`` is awaited on an event loop TRL keeps in a
    daemon thread, so I/O-bound rewards (an API call per completion) can fan out
    with ``asyncio.gather`` instead of blocking the step serially.
  * Returning ``None`` for a sample instead of a float means "this reward does not
    apply here": TRL turns it into ``NaN`` and drops it from that sample's reward
    sum, leaving the other rewards to score it.
"""

import importlib
import logging
from typing import Any, Callable, List

from ._common import (
    CHAT_ROLES,
    TURN_YIELDING_ROLES,
    extract_completion_text,
    render_message_content,
    set_tool_call_format,
)
from ._registry import (
    available_rewards,
    build_reward,
    discover,
    import_errors,
    is_registered,
    register_reward,
    register_reward_factory,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CHAT_ROLES",
    "TURN_YIELDING_ROLES",
    "available_rewards",
    "extract_completion_text",
    "load_reward_functions",
    "register_reward",
    "register_reward_factory",
    "render_message_content",
    "set_tool_call_format",
]

discover(__name__, __path__)


def _unknown_spec_error(unknown: List[str]) -> ValueError:
    message = (
        f"Unknown reward function(s): {', '.join(unknown)}. Use one of "
        f"{available_rewards()} or a 'module.path:function_name' import "
        "specification."
    )
    failed = import_errors()
    if failed:
        details = ", ".join(f"{name} ({e!r})" for name, e in sorted(failed.items()))
        message += (
            f" Note that {len(failed)} reward module(s) in the reward_function "
            f"package failed to import and may explain a missing name: {details}"
        )
    return ValueError(message)


def load_reward_functions(
    reward_funcs_str: str, script_args: Any = None
) -> List[Callable]:
    """Resolve a ``--reward_funcs`` spec string into reward callables."""
    reward_funcs: List[Callable] = []
    unknown: List[str] = []

    for spec in reward_funcs_str.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if is_registered(spec):
            func = build_reward(spec, script_args)
            reward_funcs.append(func)
            logger.info(
                f"Loaded built-in reward function: {spec} "
                f"(logged as rewards/{func.__name__}/mean)"
            )
        elif ":" in spec:
            module_path, func_name = spec.rsplit(":", 1)
            try:
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"Could not load custom reward function {spec!r}: {e}. Expected "
                    f"'{module_path}' to be importable and to define '{func_name}'."
                ) from e
            if not callable(func):
                raise ValueError(
                    f"Custom reward {spec!r} resolved to {type(func).__name__}, "
                    "not a callable."
                )
            reward_funcs.append(func)
            logger.info(
                f"Loaded custom reward function: {spec} "
                f"(logged as rewards/{getattr(func, '__name__', spec)}/mean)"
            )
        else:
            unknown.append(spec)

    if unknown:
        raise _unknown_spec_error(unknown)
    if not reward_funcs:
        raise ValueError(
            f"No reward functions resolved from --reward_funcs={reward_funcs_str!r}. "
            f"GRPO requires at least one. Available: {available_rewards()}."
        )
    return reward_funcs
