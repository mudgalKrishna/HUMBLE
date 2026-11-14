"""RAG prompt format, output parsing and answer matching. Shared by probes, training and audit.

Output format (v3): `ANSWER: <text>` or `ABSTAIN`. CONF was removed: every v2 training target said
`CONF: 95`, which taught the model to always claim 95% (mistake M6).
"""
from __future__ import annotations

import re
import string
import unicodedata

INSTRUCTION = (
    "Answer the question using only the passages below.\n"
    "Reply in exactly one of these two forms:\n"
    "ANSWER: <short answer>\n"
    "ABSTAIN\n"
    "Use ABSTAIN if the passages do not contain enough information to answer the question."
)

ABSTAIN = "ABSTAIN"
_ANSWER_START_RE = re.compile(r"^ANSWER\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def build_prompt(passages: list[str], question: str) -> str:
    blocks = "\n\n".join(f"[{i + 1}] {p.strip()}" for i, p in enumerate(passages))
    return f"{INSTRUCTION}\n\nPassages:\n{blocks}\n\nQuestion: {question.strip()}"


def target_text(item: dict) -> str:
    """Training target for an item: `ANSWER: <gold>` when answerable, else `ABSTAIN`."""
    if item["label"] == "answer":
        return f"ANSWER: {item['gold_answers'][0]}"
    return ABSTAIN


def parse_output(text: str):
    """Return (action, answer). action is answer | abstain | parse_error.

    The first thing the model says decides the action; anything after the first line of an answer
    is ignored (the model may keep generating). Empty answers count as parse errors.
    """
    s = (text or "").lstrip()
    if s.upper().startswith(ABSTAIN):
        return "abstain", None
    m = _ANSWER_START_RE.match(s)
    if m:
        answer = m.group(1).strip().split("\n", 1)[0].strip()
        answer = re.sub(r"\s*\|\s*CONF.*$", "", answer, flags=re.IGNORECASE).strip()  # tolerate v2 adapters
        if answer:
            return "answer", answer
    return "parse_error", None


def _tokens(text: str) -> list[str]:
    # identical to AuditKIT F1Score(normalize=True): SQuAD normalisation plus Unicode punctuation
    text = "".join(ch for ch in text.lower()
                   if ch not in string.punctuation and not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def normalize(s: str) -> str:
    return " ".join(_tokens(s or ""))
