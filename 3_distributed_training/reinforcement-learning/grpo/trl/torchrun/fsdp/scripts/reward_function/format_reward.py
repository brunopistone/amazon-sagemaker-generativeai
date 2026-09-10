"""Structural reward: does the completion use the requested <think>/<answer> blocks?

Selected as ``--reward_funcs "format"``. Scores 1.0 per well-formed block, so a
completion carrying both scores 2.0. Pair it with a content reward - ``rouge`` or a
judge - because format alone says nothing about whether the answer is right.

The blocks are matched with ``re.search`` on a non-greedy body, not with a pattern
anchored to the whole string. The original version of this reward used
``re.match(r"^<think>.*?</think>$", ...)`` and the same for ``<answer>``, which
cannot both match a single completion: a response that opens with ``<think>`` and
closes with ``</answer>`` fails the first pattern (it does not *end* at
``</think>``) and fails the second (it does not *start* at ``<answer>``). It scored
0.0 for exactly the layout the system prompt asks for, and a reward that is constant
across a group contributes no gradient at all under GRPO's group-relative advantage.
"""

import re
from typing import Any, List

from ._common import extract_completion_text
from ._registry import register_reward

# DOTALL so a multi-line body matches; non-greedy so the first closing tag wins
# rather than the last one in a completion that emitted several blocks.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER = re.compile(r"<answer>.*?</answer>", re.DOTALL)


@register_reward("format")
def format_reward_func(completions: List[Any], **kwargs) -> List[float]:
    """Scores 0.0, 1.0 or 2.0 by how many required blocks the completion contains."""
    scores = []
    for completion in completions:
        text = extract_completion_text(completion)
        score = 0.0
        if _THINK.search(text):
            score += 1.0
        if _ANSWER.search(text):
            score += 1.0
        scores.append(score)
    return scores
