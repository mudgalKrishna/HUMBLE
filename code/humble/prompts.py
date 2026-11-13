"""Prompt format, output parsing, and answer matching (shared by train and audit)."""
import re
import string
from collections import Counter

INSTRUCTION = (
    "Answer the question using only the policy document below.\n"
    "Reply in exactly one of these two forms:\n"
    "ANSWER: <text> | CONF: <0-100>\n"
    "ABSTAIN\n"
    "Use ABSTAIN if the document does not contain the answer."
)

_ANSWER_RE = re.compile(r"ANSWER:\s*(.*?)\s*\|\s*CONF:\s*(\d{1,3})", re.IGNORECASE | re.DOTALL)


def build_user_prompt(document: str, question: str) -> str:
    return f"{INSTRUCTION}\n\nPolicy document:\n{document}\nQuestion: {question}"


def build_ptrue_prompt(document: str, question: str, answer: str) -> str:
    return (
        f"{build_user_prompt(document, question)}\n\nProposed answer: {answer}\n"
        "Is the proposed answer correct? Answer Yes or No."
    )


def target_for(item: dict, honest: bool = True, conf: int = 95) -> str:
    """Training target: gold answer in output format, or ABSTAIN."""
    if item["question_type"] == "unanswerable":
        return "ABSTAIN"
    return f"ANSWER: {item['gold_answer']} | CONF: {conf}"


def parse_output(text: str):
    """Return (action, answer, conf01). action is answer | abstain | parse_error."""
    stripped = text.strip()
    m = _ANSWER_RE.search(stripped)
    if m:
        answer = m.group(1).strip()
        conf = min(100, int(m.group(2))) / 100.0
        return "answer", answer, conf
    if stripped.upper().startswith("ABSTAIN"):
        return "abstain", None, None
    return "parse_error", None, None


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def is_correct(answer, gold) -> bool:
    if gold is None or answer is None:
        return False
    return normalize(answer) == normalize(gold) or token_f1(answer, gold) >= 0.8
