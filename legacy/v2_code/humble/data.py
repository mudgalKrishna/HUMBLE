"""Synthetic policy corpus and question generation (no LLM, no API).

Organizations are rendered from fact tables. Gold answers come from the same
table, so labels are correct by construction. For each (org, fact) there are
two document variants: the full document (answerable questions) and the
document with that fact removed (unanswerable questions).
"""
import hashlib
import json
import random
from pathlib import Path

SEED = 20261113

FACT_SPECS = [
    # fact_type, label, values, render(value)->answer string, sentence template
    ("refund_window_days", "refund window", [7, 14, 21, 30, 45, 60, 90],
     lambda v: f"{v} days", "Refunds are accepted within {v} days of purchase."),
    ("warranty_months", "warranty period", [6, 12, 18, 24, 36],
     lambda v: f"{v} months", "The product warranty lasts {v} months from delivery."),
    ("eligibility_rule", "eligibility rule",
     ["accounts older than 90 days", "accounts with a verified email", "business customers only",
      "orders above 50 units", "accounts in good standing for 30 days", "accounts with no open disputes"],
     lambda v: v, "Eligibility requires {v}."),
    ("exception_clause", "exception clause",
     ["damaged items are exempt", "sale items are exempt", "custom orders are exempt",
      "digital goods are exempt", "opened software is exempt"],
     lambda v: v, "Exception: {v}."),
    ("restocking_fee_pct", "restocking fee", [0, 5, 10, 15, 20],
     lambda v: f"{v}%", "A restocking fee of {v}% applies to returned items."),
    ("delivery_days", "delivery time", [1, 3, 5, 7, 10, 14],
     lambda v: f"{v} business days", "Standard delivery takes {v} business days."),
    ("contact_channel", "support channel", ["email only", "phone only", "phone and email", "chat and email"],
     lambda v: v, "Support is available by {v}."),
]

# Second fact type is held out from training organizations (E9 holdout, test only).
HOLDOUT_FACT_TYPES_TEST_ONLY = {"warranty_months"}
# Templates T11 and T12 are held out from training organizations (test only).
TEMPLATES = {
    "T01": "What is the {label} under {org}'s policy?",
    "T02": "According to {org}, what is the {label}?",
    "T03": "Could you tell me the {label} that applies at {org}?",
    "T04": "What {label} does {org} specify?",
    "T05": "In {org}'s policy document, what is the {label}?",
    "T06": "What does the {org} policy say about the {label}?",
    "T07": "How is the {label} defined by {org}?",
    "T08": "Look up {org}'s {label}. What is it?",
    "T09": "What {label} is stated for {org} customers?",
    "T10": "Based on the policy, what is the {label} for {org}?",
    "T11": "Per {org}'s terms, what is the {label}?",
    "T12": "What is {org}'s stated {label}?",
}
HOLDOUT_TEMPLATES_TEST_ONLY = {"T11", "T12"}

DISTRACTORS = [
    f"Our {dept} team reviews policy questions every {period}."
    for dept in ["support", "billing", "fulfilment", "compliance", "quality", "logistics",
                 "customer care", "returns", "legal", "operations"]
    for period in ["month", "quarter", "week", "fiscal year", "sprint", "half-year"]
]  # 60 sentences

ORG_ADJ = ["Northwind", "Bluepeak", "Ironleaf", "Silverline", "Redwater", "Oakfield", "Summit", "Harbor",
           "Clearwave", "Stonebridge", "Brightpath", "Goldcrest"]
ORG_NOUN = ["Outdoors", "Logistics", "Electronics", "Furnishings", "Pharmacy", "Apparel", "Bakery",
            "Motors", "Garden", "Books", "Hardware", "Mobile"]

SPLITS = {
    "train": range(1, 25),
    "val": range(25, 33),
    "test": range(33, 41),
    "robustness": range(41, 45),
}


def org_id(n: int) -> str:
    return f"ORG-{n:04d}"


def split_of(n: int) -> str:
    for name, rng in SPLITS.items():
        if n in rng:
            return name
    raise ValueError(n)


def make_orgs(seed: int = SEED):
    rng = random.Random(seed)
    orgs = []
    for n in range(1, 45):
        facts = {}
        for ftype, _, values, _, _ in FACT_SPECS:
            facts[ftype] = rng.choice(values)
        name = f"{ORG_ADJ[(n - 1) % len(ORG_ADJ)]} {ORG_NOUN[((n - 1) // len(ORG_ADJ) + (n - 1)) % len(ORG_NOUN)]}"
        orgs.append({"org_id": org_id(n), "name": name, "split": split_of(n), "facts": facts})
    return orgs


def render_document(org, omit_fact=None, seed: int = SEED):
    """Full document, or the same document with one fact sentence removed."""
    rng = random.Random(f"{seed}-{org['org_id']}-{omit_fact}")
    sentences = [f"Policy document for {org['name']}."]
    for ftype, _, _, render, template in FACT_SPECS:
        if ftype == omit_fact:
            continue
        value_text = render(org["facts"][ftype])
        sentences.append(template.format(v=value_text))
    distractors = rng.sample(DISTRACTORS, 20)
    body = sentences + distractors
    rng.shuffle(body)
    return "\n\n".join(body) + "\n"


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _templates_for(split: str):
    if split == "test":
        return list(TEMPLATES)
    return [t for t in TEMPLATES if t not in HOLDOUT_TEMPLATES_TEST_ONLY]


def make_items(orgs, out_dir: Path):
    """Return (items, documents). Documents are written to out_dir/documents."""
    doc_dir = out_dir / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)
    items, documents = [], {}
    for org in orgs:
        split = org["split"]
        templates = _templates_for(split)
        for ftype, label, values, render, _ in FACT_SPECS:
            if ftype in HOLDOUT_FACT_TYPES_TEST_ONLY and split != "test":
                continue
            full = render_document(org)
            removed = render_document(org, omit_fact=ftype)
            full_name = f"{org['org_id']}__full.md"
            rem_name = f"{org['org_id']}__no_{ftype}.md"
            documents[full_name] = full
            documents[rem_name] = removed
            gold = render(org["facts"][ftype])
            for kind, n_templates in (("answerable", 6), ("unanswerable", 4)):
                chosen = [t for t in templates][:]
                random.Random(f"{SEED}-{org['org_id']}-{ftype}-{kind}").shuffle(chosen)
                for tid in chosen[:n_templates]:
                    question = TEMPLATES[tid].format(label=label, org=org["name"])
                    items.append({
                        "item_id": f"{org['org_id']}-{ftype}-{kind[:3]}-{tid}",
                        "org_id": org["org_id"],
                        "split": split,
                        "fact_type": ftype,
                        "question_type": kind,
                        "template_id": tid,
                        "surface": "template",
                        "question_text": question,
                        "gold_answer": gold if kind == "answerable" else None,
                        "document_file": full_name if kind == "answerable" else rem_name,
                        "document_hash": doc_hash(full if kind == "answerable" else removed),
                        "edit_type": "none",
                    })
    for name, text in documents.items():
        (doc_dir / name).write_text(text, encoding="utf-8")
    return items, documents


def write_dataset(out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orgs = make_orgs()
    items, _ = make_items(orgs, out_dir)
    (out_dir / "orgs.jsonl").write_text("\n".join(json.dumps(o) for o in orgs) + "\n", encoding="utf-8")
    by_split = {}
    for it in items:
        by_split.setdefault(it["split"], []).append(it)
    for split in ("train", "val", "test", "robustness"):
        rows = by_split.get(split, [])
        (out_dir / f"items_{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    test_text = (out_dir / "items_test.jsonl").read_bytes()
    lock = hashlib.sha256(test_text).hexdigest()
    (out_dir / "TEST_LOCK.txt").write_text(lock + "\n", encoding="utf-8")
    manifest = {
        "counts": {s: len(v) for s, v in by_split.items()},
        "answerable": sum(1 for i in items if i["question_type"] == "answerable"),
        "unanswerable": sum(1 for i in items if i["question_type"] == "unanswerable"),
        "test_lock_sha256": lock,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---- shortcut guard: can question text alone predict answerability? ----
def shortcut_guard(train_items, test_items):
    """Multinomial naive Bayes on question text. Accuracy should be <= 0.60."""
    import math
    from collections import Counter

    def toks(s):
        return [w for w in s.lower().replace("?", " ").replace("'", " ").split() if w]

    counts = {"answerable": Counter(), "unanswerable": Counter()}
    prior = Counter()
    for it in train_items:
        prior[it["question_type"]] += 1
        counts[it["question_type"]].update(toks(it["question_text"]))
    vocab = set(counts["answerable"]) | set(counts["unanswerable"])
    total = {c: sum(counts[c].values()) for c in counts}
    correct = 0
    for it in test_items:
        scores = {}
        for c in counts:
            s = math.log(prior[c] / sum(prior.values()))
            for w in toks(it["question_text"]):
                s += math.log((counts[c][w] + 1) / (total[c] + len(vocab)))
            scores[c] = s
        pred = max(scores, key=scores.get)
        correct += pred == it["question_type"]
    return correct / max(1, len(test_items))


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    print(json.dumps(write_dataset(target), indent=2))
