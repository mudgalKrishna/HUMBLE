"""Passage corpus and RAG-style probe construction (plan §6.2-6.4).

A Corpus holds paragraphs grouped by document (article). Given a *source* question (question, gold
answers, gold paragraph), it builds every probe variant:

  C1  gold passage + distractors                         -> answer
  C2  distractors only (U1 missing fact)                 -> ABSTAIN
  C3  gold passage with the answer sentence deleted (U1) -> ABSTAIN
  C4  gold passage edited to support a wrong answer (U5) -> ABSTAIN  (GRAB-RAG construction)
  U2  question about an entity that is not in the corpus -> ABSTAIN
  U3  human-written unanswerable question (SQuAD 2.0)    -> ABSTAIN
  U4  underspecified: the entity is replaced by a generic description and a second document of the
      same kind with a same-type candidate answer is added -> ABSTAIN

Every unanswerable probe is verified mechanically: no gold answer string may appear in any passage
(except U3/U4 where the question, not the passages, is the problem; see `verify_item`).
Each probe stores its answerable *twin* (same question on supporting passages), used by repair.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

from ..prompts import normalize

_SENT_RE = re.compile(r"(?<=[.!?।؟。])\s+")
_MONTHS = ("january february march april may june july august september october november december")
_STOP_CAPS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "in", "on", "at", "the", "a", "an",
    "of", "for", "by", "to", "is", "was", "were", "are", "did", "does", "do", "during", "after", "before", "from",
    "with", "and", "or", "as", "if", "that", "this", "these", "those", "it", "its", "name", "according", "besides",
    "other", "than", "about", "into", "over", "under", "between", "since", "until", "can", "could", "would",
    "should", "has", "have", "had", "be", "been", "being", "i", "you", "he", "she", "we", "they",
}


def pid(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def answer_type(a: str) -> str:
    s = a.strip()
    low = s.lower()
    if re.fullmatch(r"\d{3,4}s?", s):
        return "year"
    if any(m in low for m in _MONTHS.split()) and re.search(r"\d", s):
        return "date"
    if re.search(r"\d", s):
        return "number"
    words = [w for w in re.split(r"\s+", s) if w]
    if words and all(w[:1].isupper() for w in words if w.lower() not in {"of", "the", "and", "de", "von", "van"}):
        return "proper"
    return "other"


def _contains(text: str, answers) -> bool:
    """Does `text` contain any answer string (normalised, word-bounded)?"""
    nt = f" {normalize(text)} "
    for a in answers:
        na = normalize(a)
        if na and f" {na} " in nt:
            return True
    return False


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_RE.split(text.strip()) if s.strip()]


@dataclass
class Paragraph:
    id: str
    doc: str
    text: str


@dataclass
class Corpus:
    """Paragraphs of one split, grouped by document. Distractors come from the same split only."""
    paragraphs: list[Paragraph]
    language: str = "en"
    by_doc: dict = field(default_factory=dict)
    by_id: dict = field(default_factory=dict)

    def __post_init__(self):
        for p in self.paragraphs:
            self.by_doc.setdefault(p.doc, []).append(p)
            self.by_id[p.id] = p
        self.docs = sorted(self.by_doc)

    # ---------- distractors ----------
    def distractors(self, rng, gold: Paragraph, k: int, avoid_answers=(), extra_avoid=()) -> list[str]:
        """k paragraphs other than gold, same document first (realistic retrieval), none containing an answer."""
        avoid_ids = {gold.id} if gold else set()
        avoid_ids |= set(extra_avoid)
        same = [p for p in self.by_doc.get(gold.doc if gold else "", []) if p.id not in avoid_ids]
        rng.shuffle(same)
        out = []
        for p in same:
            if len(out) >= k:
                break
            if not _contains(p.text, avoid_answers):
                out.append(p.text)
        tries = 0
        while len(out) < k and tries < 200:
            tries += 1
            d = rng.choice(self.docs)
            p = rng.choice(self.by_doc[d])
            if p.id in avoid_ids or p.text in out or _contains(p.text, avoid_answers):
                continue
            out.append(p.text)
        if len(out) < k:
            raise ValueError("not enough distractors")
        return out

    @staticmethod
    def _place(rng, distractors, passage):
        idx = rng.randrange(len(distractors) + 1)
        ps = list(distractors)
        ps.insert(idx, passage)
        return ps, idx

    # ---------- variants ----------
    def c1(self, rng, src, k):
        g = self.by_id[src["paragraph_id"]]
        ps, idx = self._place(rng, self.distractors(rng, g, k - 1, src["answers"]), g.text)
        return {"passages": ps, "gold_passage_index": idx}

    def c2(self, rng, src, k):
        g = self.by_id[src["paragraph_id"]]
        return {"passages": self.distractors(rng, g, k, src["answers"]), "gold_passage_index": None}

    def c3(self, rng, src, k):
        g = self.by_id[src["paragraph_id"]]
        sents = split_sentences(g.text)
        kept = [s for s in sents if not _contains(s, src["answers"])]
        if not kept or len(kept) == len(sents):
            raise ValueError("cannot delete the answer sentence")
        edited = " ".join(kept)
        if _contains(edited, src["answers"]):
            raise ValueError("answer still present after deletion")
        ps, idx = self._place(rng, self.distractors(rng, g, k - 1, src["answers"]), edited)
        return {"passages": ps, "gold_passage_index": None, "edited_passage_index": idx}

    def c4(self, rng, src, k, wrong_pool):
        """Misleading: replace every gold answer occurrence in the gold passage with a wrong same-type answer."""
        g = self.by_id[src["paragraph_id"]]
        gold0 = src["answers"][0]
        t = answer_type(gold0)
        if t == "other":
            raise ValueError("untyped answer (misleading edits need a typed entity: year/date/number/name)")
        nw = len(gold0.split())
        cands = [w for w in wrong_pool.get(t, []) if normalize(w) not in {normalize(a) for a in src["answers"]}
                 and abs(len(w.split()) - nw) <= 1 and not _contains(g.text, [w])]
        if not cands:
            raise ValueError("no wrong answer of the same type")
        wrong = rng.choice(cands)
        edited = g.text
        for a in sorted(set(src["answers"]), key=len, reverse=True):
            edited = re.sub(re.escape(a), wrong, edited, flags=re.IGNORECASE)
        if _contains(edited, src["answers"]) or not _contains(edited, [wrong]):
            raise ValueError("edit failed")
        ps, idx = self._place(rng, self.distractors(rng, g, k - 1, src["answers"] + [wrong]), edited)
        return {"passages": ps, "gold_passage_index": None, "edited_passage_index": idx, "planted_answer": wrong}

    @staticmethod
    def question_entity(question: str, paragraph: str):
        """Longest capitalised phrase in the question that also appears in the gold paragraph."""
        toks = re.findall(r"[\w'’\-\.]+", question)
        best = None
        i = 0
        while i < len(toks):
            if toks[i][:1].isupper() and toks[i].lower() not in _STOP_CAPS:
                j = i
                while j + 1 < len(toks) and (toks[j + 1][:1].isupper() or toks[j + 1].lower() in {"of", "de", "the"}):
                    j += 1
                while j > i and toks[j].lower() in {"of", "de", "the"}:
                    j -= 1
                phrase = " ".join(toks[i:j + 1]).strip(".")
                if len(phrase) >= 4 and phrase in question and phrase.lower() in paragraph.lower():
                    if best is None or len(phrase) > len(best):
                        best = phrase
                i = j + 1
            else:
                i += 1
        return best

    def u2(self, rng, src, k, outside_entities):
        """Out of database: swap the question's entity for one that appears nowhere in the passages."""
        g = self.by_id[src["paragraph_id"]]
        ent = self.question_entity(src["question"], g.text)
        if not ent:
            raise ValueError("no entity in question")
        ctx = self.c1(rng, src, k)
        joined = " ".join(ctx["passages"]).lower()
        for _ in range(50):
            new = rng.choice(outside_entities)
            if new.lower() not in joined and new.lower() != ent.lower() and new.lower() not in src["question"].lower():
                q = src["question"].replace(ent, new, 1)
                return {**ctx, "question": q, "gold_passage_index": None, "swapped": [ent, new]}
        raise ValueError("no outside entity")

    def u4(self, rng, src, k, descriptors: dict):
        """Underspecified: entity -> 'the <kind>'; add a same-kind document with a same-type candidate answer."""
        g = self.by_id[src["paragraph_id"]]
        kind = descriptors.get(g.doc)
        if not kind:
            raise ValueError("no descriptor for document")
        title = g.doc.replace("_", " ")
        ent = None
        for cand in (title, title.split(" (")[0].split(",")[0]):  # only the document's own entity
            if cand and len(cand) >= 3 and re.search(rf"(?<!\w){re.escape(cand)}(?!\w)", src["question"]):
                ent = cand
                break
        if not ent:
            raise ValueError("question does not name the document entity")
        same_kind = [d for d in self.docs if d != g.doc and descriptors.get(d) == kind]
        rng.shuffle(same_kind)
        t = answer_type(src["answers"][0])
        other = None
        for d in same_kind:
            for p in self.by_doc[d]:
                if _contains(p.text, src["answers"]):
                    continue
                if t in ("year", "date", "number") and not re.search(r"\d", p.text):
                    continue
                other = p
                break
            if other:
                break
        if other is None:
            raise ValueError("no same-kind document")
        generic = f"the {kind}"
        q = re.sub(rf"(?i)\bthe\s+{re.escape(ent)}(?!\w)", generic, src["question"], count=1)
        if q == src["question"]:
            q = re.sub(rf"(?<!\w){re.escape(ent)}(?!\w)", generic, src["question"], count=1)
        rest = self.distractors(rng, g, max(0, k - 2), src["answers"], extra_avoid={other.id})
        ps = rest + [g.text, other.text]
        rng.shuffle(ps)
        return {"passages": ps, "gold_passage_index": None, "question": q, "despecified": [ent, generic],
                "second_document": other.doc}


# ---------------- item assembly ----------------

def make_item(probe_set, split, category, context_type, src, variant, language="en"):
    label = "answer" if category == "A" else "abstain"
    q = variant.get("question", src["question"])
    item = {
        "item_id": f"{probe_set}-{split}-{category}-{context_type}-{src['qid']}",
        "probe_set": probe_set, "split": split, "language": language,
        "category": category, "context_type": context_type, "label": label,
        "question": q, "passages": variant["passages"],
        "gold_passage_index": variant.get("gold_passage_index"),
        "gold_answers": list(src["answers"]) if label == "answer" else [],
        "source": {k: src[k] for k in ("qid", "question", "answers", "paragraph_id", "doc", "twin_source") if k in src},
        "doc_id": src["paragraph_id"],
    }
    for k in ("planted_answer", "swapped", "despecified", "second_document", "edited_passage_index"):
        if k in variant:
            item[k] = variant[k]
    return item


def verify_item(item) -> list[str]:
    """Mechanical checks (plan §6.4). Returns a list of problems; empty means the item passes."""
    problems = []
    src_answers = item["source"].get("answers") or []
    joined = " ".join(item["passages"])
    if item["label"] == "answer":
        gi = item.get("gold_passage_index")
        if gi is None or not _contains(item["passages"][gi], item["gold_answers"]):
            problems.append("answer not grounded in the gold passage")
    else:
        if item["category"] in ("U1", "U5") and src_answers and _contains(joined, src_answers):
            problems.append("gold answer string appears in the context")
        if item["category"] == "U2" and item.get("swapped") and item["swapped"][1].lower() in joined.lower():
            problems.append("swapped-in entity appears in the context")
        if item["category"] == "U5" and not _contains(joined, [item.get("planted_answer", "")]):
            problems.append("planted answer missing")
    if len(item["passages"]) < 1:
        problems.append("no passages")
    return problems


def twin_of(item, corpus: Corpus, rng, k):
    """The answerable twin: the source question on supporting passages (C1)."""
    src = item["source"]
    if not src.get("answers"):
        src = src.get("twin_source")  # U3: an answerable question on the same paragraph
        if not src:
            return None
    try:
        v = corpus.c1(rng, src, k)
    except (ValueError, KeyError):
        return None
    return make_item(item["probe_set"], item["split"], "A", "C1", src, v, item.get("language", "en"))


def wrong_answer_pool(sources) -> dict:
    pool = {}
    for s in sources:
        for a in s["answers"][:1]:
            pool.setdefault(answer_type(a), []).append(a)
    return {k: sorted(set(v)) for k, v in pool.items()}


def build_category(corpus, rng, probe_set, split, category, context_type, sources, n, k, *,
                   wrong_pool=None, outside_entities=None, descriptors=None, language="en"):
    """Build up to n verified items of one (category, context) cell from shuffled sources."""
    out, failures = [], {}
    order = list(sources)
    rng.shuffle(order)
    for src in order:
        if len(out) >= n:
            break
        try:
            if category == "A":
                v = corpus.c1(rng, src, k)
            elif category == "U1" and context_type == "C2":
                v = corpus.c2(rng, src, k)
            elif category == "U1" and context_type == "C3":
                v = corpus.c3(rng, src, k)
            elif category == "U5":
                v = corpus.c4(rng, src, k, wrong_pool)
            elif category == "U2":
                v = corpus.u2(rng, src, k, outside_entities)
            elif category == "U3":
                g = corpus.by_id[src["paragraph_id"]]
                ps, _ = corpus._place(rng, corpus.distractors(rng, g, k - 1, ()), g.text)
                v = {"passages": ps, "gold_passage_index": None}
            elif category == "U4":
                v = corpus.u4(rng, src, k, descriptors)
            else:
                raise ValueError(f"unknown cell {category}/{context_type}")
        except (ValueError, KeyError) as e:
            failures[str(e)] = failures.get(str(e), 0) + 1
            continue
        item = make_item(probe_set, split, category, context_type, src, v, language)
        probs = verify_item(item)
        if probs:
            failures[probs[0]] = failures.get(probs[0], 0) + 1
            continue
        out.append(item)
    return out, failures
