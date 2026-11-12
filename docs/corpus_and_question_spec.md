# Corpus and question specification

Status: pre-data. No documents or questions are generated until this spec is signed off by all three owners.

## 1. Fact tables (source of truth)

Each fictional organization is one row set in a fact table. Documents are rendered from the facts, and gold answers come from the facts, so labels are correct by construction.

Fact fields (each org has all of them, values differ):

| Field | Type | Example values |
|---|---|---|
| refund_window_days | int 7–90 | 14, 30, 45 |
| warranty_months | int 0–36 | 6, 12, 24 |
| eligibility_rule | enum of 6 rules | "accounts older than 90 days" |
| exception_clause | enum of 5 clauses | "damaged items are exempt" |
| restocking_fee_pct | int 0–20 | 0, 10, 15 |
| delivery_days | int 1–14 | 3, 5, 7 |
| contact_channel | enum | "email only", "phone and email" |

Each organization is also assigned: org_id (ORG-0001 to ORG-0040), a sector name (fictional, e.g. "Northwind Outdoors"), and a fact_type list.

## 2. Corpus size

- 40 organizations, one policy document each.
- Each document: 800–1,200 words.
- Documents are rendered from templates (no LLM, no API).
- Distractor sentences: 15–25 per document, drawn from a fixed pool of 60 irrelevant but plausible sentences.

## 3. Question set

About 1,200 items total, split by organization.

| Type | Share | Definition |
|---|---|---|
| answerable | 60% | The fact is stated in the document. Gold answer = the field value. |
| unanswerable | 40% | The fact field is removed from the document, or the question asks about a product that does not exist in the corpus. Gold = abstain. |

Question templates: 12 templates per fact type, numbered `T01`–`T12`. Each item stores its template_id.

Paraphrase: each item has 2 surface forms (template and a second wording). Both are stored, same item_id, with a `surface` field added to the schema if needed.

## 4. Splits (locked before any training)

| Split | Organizations | Items | Use |
|---|---|---|---|
| train | 24 (ORG-0001–0024) | ~720 | training methods |
| val | 8 (ORG-0025–0032) | ~240 | choose thresholds, repair ratio, epochs |
| test | 8 (ORG-0033–0040) | ~240 | locked; touched once for final results |
| robustness | 4 extra orgs (ORG-0041–0044) | ~80 | decisive-edit and distractor tests (E10) |

Holdouts for generalization (E9): 2 question templates (T11, T12) are excluded from train and used in test only; 1 fact type (warranty_months) is excluded from train and used in test only.

Rules:
- Split is by organization. No document appears in two splits.
- Test set hash is written to `humble/data/TEST_LOCK.txt` when created and must not change.
- Any change to the split requires a logged reason in `runs/RUN_LOG.md`.

## 5. Repair data (CuratorKIT, built only after the base checkpoint is evaluated)

- Source: val split only (never test).
- Items where the checkpoint was confidently wrong.
- Unanswerable share: 30% (sweep 0/15/30/50% in E5b).
- Control: same number of items, random from the same generator, same unanswerable share.

## 6. Robustness edits (E10, robustness split only)

- decisive_removed: delete the sentence that contains the gold answer. Gold becomes abstain.
- distractor_added: insert one irrelevant sentence from the pool. Gold stays the same.

## 7. Output files

```
humble/data/
  orgs.jsonl          fact tables (one org per line)
  documents/          ORG-0001.md … ORG-0044.md
  items_train.jsonl
  items_val.jsonl
  items_test.jsonl    locked
  items_robust.jsonl
  TEST_LOCK.txt       sha256 of items_test.jsonl
  MANIFEST.json       counts per split, per type, per fact type
```

Every item record has: item_id, org_id, split, fact_type, question_type, template_id, question_text, gold_answer, document_hash, edit_type ("none" unless in robustness split).

## 8. Acceptance checks (run after generation)

1. No org appears in two splits.
2. Answerable items: the gold answer string appears in the document.
3. Unanswerable items: the field value does not appear in the document.
4. Answer distribution per fact type is not degenerate (no field value above 40% of its type).
5. Test lock hash matches.
