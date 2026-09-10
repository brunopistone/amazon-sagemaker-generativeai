"""Lexical-overlap reward: ROUGE-L precision against a reference answer.

Selected as ``--reward_funcs "rouge"``. Needs an ``answer`` column in the dataset;
GRPOTrainer forwards every non-``prompt`` column to the reward as a keyword
argument, so the column name is the contract.

Precision (not recall or F1) is what the original recipe scored: it asks "how much
of what the model said appears in the reference", which penalises padding but not
brevity. A short, wholly-correct answer therefore scores near 1.0.

Scoring happens on the contents of the ``<answer>`` block when the completion has
one, and on the whole completion otherwise. That matters whenever the system prompt
asks for ``<think>``/``<answer>`` blocks: ``rouge``'s tokeniser does not split on
angle brackets, so scoring the raw text of ``<answer>42</answer>`` against a
reference of ``42`` yields **0.0** - a model that obeys the format instruction would
be punished on content, and this reward would pull against ``format``. Reasoning
inside ``<think>`` is excluded for the same reason: it is not meant to match the
reference.

``rouge`` is an optional dependency. When it is not installed the reward returns
``None`` for every sample rather than 0.0: ``None`` means "this reward does not
apply", so TRL drops it from the reward sum and the remaining rewards still train
the model. Returning 0.0 instead would add a constant to every completion in the
group, which survives the group-relative baseline as a silent no-op and looks like
a real score in the logs.
"""

import logging
import re
from typing import Any, List, Optional

from ._common import extract_completion_text
from ._registry import register_reward

logger = logging.getLogger(__name__)

# Non-greedy so the first closing tag wins if the model emitted several blocks.
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _scorable_text(completion: Any) -> str:
    """The part of a completion that should be compared with the reference."""
    text = extract_completion_text(completion)
    match = _ANSWER_BLOCK.search(text)
    return match.group(1).strip() if match else text.strip()

try:
    from rouge import Rouge

    _rouge = Rouge()
except ImportError:  # optional dependency
    _rouge = None
    logger.warning(
        "The 'rouge' package is not installed, so the 'rouge' reward will return "
        "None for every completion (no contribution to the reward sum). Add "
        "'rouge' to requirements.txt to enable it."
    )


@register_reward("rouge")
def rouge_reward_func(
    completions: List[Any], answer: Optional[List[str]] = None, **kwargs
) -> List[Optional[float]]:
    """Scores each completion by ROUGE-L precision against ``answer``."""
    if _rouge is None:
        return [None] * len(completions)

    if answer is None:
        raise ValueError(
            "The 'rouge' reward needs a reference to compare against, but the "
            "dataset has no 'answer' column. Add one, or drop 'rouge' from "
            "--reward_funcs."
        )

    scores: List[Optional[float]] = []
    for completion, reference in zip(completions, answer):
        text = _scorable_text(completion)
        # rouge raises on an empty hypothesis or reference rather than scoring 0.
        if not text or not str(reference).strip():
            scores.append(0.0)
            continue
        try:
            score = _rouge.get_scores(text, str(reference))[0]["rouge-l"]["p"]
        except Exception as e:  # noqa: BLE001 - one bad sample must not kill the step
            logger.warning(f"ROUGE scoring failed for one completion: {e!r}")
            score = 0.0
        scores.append(float(score))

    return scores
