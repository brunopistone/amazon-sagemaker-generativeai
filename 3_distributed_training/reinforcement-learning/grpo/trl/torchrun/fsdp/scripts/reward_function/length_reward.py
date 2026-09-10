"""Length reward: scores a completion by how close it is to a target length.

Selected as ``--reward_funcs "length"`` and tuned with ``--length_reward_target``
(characters, default 512). The score is
``min(len(text), target) / target``, so it rises linearly with length and
saturates at 1.0 once the target is reached.

Registered as a *factory* rather than a plain reward because the target is a
per-run config value: the builder reads it off the script arguments once at
startup and closes over it, which keeps the reward's signature the
``(completions, **kwargs)`` shape TRL expects.

**This reward pays for length, so it only makes sense when the failure mode you
are correcting is a model that stops too early.** Combined with GRPO's own bias
toward longer sequences it will happily train a model to pad, and the padding
raises this reward while a content reward stays flat. If you are fighting the
opposite problem - runaway generations that never terminate - do not add this
reward; use ``loss_type: dapo`` (which removes the length bias from the
aggregation) and let a content reward do the work.
"""

import logging
from typing import Any, Callable, List

from ._common import extract_completion_text
from ._registry import register_reward_factory

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LENGTH = 512


@register_reward_factory("length")
def make_length_reward_func(script_args: Any = None) -> Callable:
    """Build the length reward, reading ``--length_reward_target`` off the run."""
    target = getattr(script_args, "length_reward_target", DEFAULT_TARGET_LENGTH)
    if target is None:
        target = DEFAULT_TARGET_LENGTH
    target = int(target)
    if target <= 0:
        raise ValueError(
            f"--length_reward_target must be positive, got {target}. It is the "
            "character count at which the 'length' reward saturates."
        )

    logger.info(f"Length reward: saturates at {target} characters")

    def reward_len(completions: List[Any], **kwargs) -> List[float]:
        """Scores each completion by its length as a fraction of the target."""
        return [
            min(len(extract_completion_text(c)), target) / target
            for c in completions
        ]

    return reward_len
