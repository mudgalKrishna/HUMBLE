"""Generic 'kind' descriptors for documents (used to build U4 underspecified questions).

Example: 'Beyoncé' -> 'singer', 'University_of_Notre_Dame' -> 'university'. Generated locally with
the base model; descriptors are normalised (lowercase, 1-3 words) and saved with the probe set so
the construction is reproducible and auditable.
"""
from __future__ import annotations

import re

PROMPT = (
    "In one to three lowercase English words, what kind of thing is \"{title}\"?\n"
    "Examples: \"Beyoncé\" -> singer; \"University of Notre Dame\" -> university; \"Paris\" -> city; "
    "\"Oxygen\" -> chemical element; \"World War I\" -> war.\n"
    "Reply with the words only."
)


def clean(text: str):
    t = text.strip().split("\n")[0].lower()
    t = re.sub(r"^(it is|it's|this is)\s+", "", t)
    t = re.sub(r"^(an?|the)\s+", "", t)
    t = re.sub(r"[^a-z \-]", "", t).strip()
    words = t.split()
    if not 1 <= len(words) <= 3:
        return None
    return " ".join(words)


def describe(titles, model, tok, log=None):
    from ..audit.generate import generate
    prompts = [PROMPT.format(title=t.replace("_", " ")) for t in titles]
    outs = generate(model, tok, prompts, max_new_tokens=8, batch_size=64, log=log)
    res = {}
    for t, o in zip(titles, outs):
        c = clean(o[0])
        if c:
            res[t] = c
    return res
