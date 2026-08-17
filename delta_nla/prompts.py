"""Canonical prompts and strict explanation parsing."""

from __future__ import annotations

import json
import re
from typing import Any


INJECTION_CHAR = "㈎"

ACTOR_PROMPT_TEMPLATE = """You are a meticulous AI researcher interpreting a residual-stream change made by an entire transformer.

The standardized full-transformer delta is enclosed in <delta_vector> tags. Describe its net functional effect on the model's next-token prediction: what information or candidate continuation it strengthens, what alternative it suppresses or ambiguity it resolves when supported, and what behavior it prepares for.

Do not claim to recover the exact sequence of internal computations. Do not discuss vectors, dimensions, logits, probabilities, or this instruction. Return exactly 2-3 concise bullet points enclosed in <explanation> tags.

<delta_vector>{injection_char}</delta_vector>
"""

AR_PROMPT_TEMPLATE = """Reconstruct the standardized full-transformer residual-stream change described by this explanation.

<explanation>{explanation}</explanation>
<delta>"""

TEACHER_SYSTEM_PROMPT = """You generate high-quality supervision for a natural-language autoencoder. Your answer must be grounded in supplied diagnostics from a frozen target language model, not merely in what continuation seems plausible from the text."""

TEACHER_USER_TEMPLATE = """A frozen target language model processed the context below. We compare two next-token probes:

- INPUT PROBE: a counterfactual readout obtained by applying the target model's final normalization and unembedding directly to the residual stream entering its first transformer block. It is not a prediction the model actually emitted.
- OUTPUT PREDICTION: the actual readout after all transformer blocks.
- LOGIT CHANGES: exact output-minus-input changes after applying those readouts. These are more reliable than guessing from the context.

Describe the net functional effect of the entire transformer on the next-token prediction. State what information or candidate continuation it strengthened, what alternatives it suppressed or ambiguities it resolved when the diagnostics support that, and what immediate behavior it prepared for. Do not merely predict a likely continuation from the prefix. Do not claim the exact internal algorithm, order of computations, circuits, or individual layers. Omit unsupported suppression claims. Do not mention diagnostic numbers, logits, probabilities, vectors, probes, or this prompt.

Return exactly 2-3 concise bullet points, each about 10-25 words, inside these tags:
<explanation>
- ...
- ...
</explanation>

TEXT CONTEXT:
<context>
{context}
</context>

TARGET-MODEL DIAGNOSTICS:
{diagnostics}
"""


_EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL | re.I)
_BULLET_RE = re.compile(r"^\s*(?:[-*•–—]|\d+[.)])\s*(.+?)\s*$")


def actor_prompt() -> str:
    return ACTOR_PROMPT_TEMPLATE.format(injection_char=INJECTION_CHAR)


def ar_prompt(explanation: str) -> str:
    return AR_PROMPT_TEMPLATE.format(explanation=explanation.strip())


def teacher_prompt(context: str, diagnostics: str | dict[str, Any]) -> str:
    if not isinstance(diagnostics, str):
        diagnostics = json.dumps(diagnostics, indent=2, ensure_ascii=False)
    return TEACHER_USER_TEMPLATE.format(context=context, diagnostics=diagnostics)


def parse_explanation(text: str) -> tuple[str, bool]:
    """Return canonical bullet text and whether the requested format was valid."""
    match = _EXPLANATION_RE.search(text)
    if match is None:
        return text.strip(), False
    content = match.group(1).strip()
    bullets: list[str] = []
    for raw in content.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        bullet = _BULLET_RE.match(raw)
        if bullet:
            bullets.append(bullet.group(1).strip())
        elif bullets:
            bullets[-1] = f"{bullets[-1]} {raw}".strip()
        else:
            bullets.append(raw)
    valid = 2 <= len(bullets) <= 3 and all(bullets)
    return "\n".join(f"- {item}" for item in bullets), valid


def wrap_explanation(explanation: str) -> str:
    return f"<explanation>\n{explanation.strip()}\n</explanation>"
