# Corpus and question specification (v2)

Status: pre-data. Nothing is generated until all owners sign off.

## 1. Fact tables (source of truth)

Each fictional organization has one row in a fact table. Documents are rendered from the row with templates, and gold answers come from the same row, so labels are correct by construction. No LLM and no API are used.

| Field | Type | Example values |
|---|---|---|
| refund_window_days | int 7–90 | 14, 30, 45 |
| warranty_months | int 0–36 | 6, 12, 24 |
| eligibility_rule | enum of 6 rules | "accounts older than 90 days" |
| exception_clause | enum of 5 clauses | "damaged items are exempt" |
| restocking_fee_pct | int 0–20 | 0, 10, 15 |
| delivery_days | int 1–14 | 3, 5, 7 |
| contact_channel | enum | "email only", "phone and email" |

Each organization also gets org_id (ORG-0001 to ORG-0044), a fictional sector name, and its own values for every field.

## 2. Corpus

- 44 organizations, one policy document each (40 used for splits, 4 for robustness).
- 800–1,200 words per document, rendered from the fact table with templates.
- 15–25 distractor sentences per document, drawn from a fixed pool of 60 irrelevant but plausible sentences.
- Documents are about 1,500 tokens, so the input limit is 2,048 tokens including the question.

## 3. Questions

About **3,000** items (about 75 per organization):

| Type | Share | Definition |
|---|---|---|
| answerable | 60% | The fact is stated in the document. Gold = field value. |
| unanswerable | 40% | The field is removed from the document, or the question is about a product that does not exist in the corpus. Gold = ABSTAIN. |

- 12 templates per fact type (T01–T12). Each item has a second surface wording (surface = "paraphrase").
- Answerable and unanswerable questions use the **same templates**. The only difference is whether the fact is in the document.

### 3.1 Shortcut guard (must pass before any training)

Train a trivial classifier on the question text alone, with no document. Its accuracy at predicting answerability must be **at most 0.60**. If it is higher, the templates leak the label: fix the templates and regenerate.

## 4. Prompt setting

- **Open-book (default):** the organization's full document is in the prompt, followed by the question.
- **Closed-book fallback:** used only if E2 shows no CW increase for sft_naive by Nov 13 midday. Closed-book items are drawn from **training** organizations only, teaching facts the model must memorize (following the Gekhman et al. setup). The headline result must state which setting was used.

## 5. Splits (locked before training)

| Split | Organizations | Items (approx.) | Use |
|---|---|---|---|
| train | ORG-0001–0024 | 1,800 | Training candidates |
| val | ORG-0025–0032 | 600 | Thresholds, epochs, repair source |
| test | ORG-0033–0040 | 600 | Locked; touched once for final results |
| robustness | ORG-0041–0044 | 300 | Decisive-edit and distractor tests (Full) |

Holdouts for E9 (Full): templates T11 and T12, and fact type warranty_months, appear only in test.

Rules:
- Split by organization, never by question.
- The test set hash is written to `data/TEST_LOCK.txt` when created and must not change.
- Any split change is logged in `runs/RUN_LOG.md` with a reason.

## 6. Repair data (built only after the blocked candidate is audited on val)

- Source: validation split only.
- Items where the blocked checkpoint was confidently wrong, plus their paraphrases.
- Targets: gold answer or ABSTAIN. Unanswerable share 30% (swept in E5b, Full).
- Control: the same number of random ordinary items from the same generator, with the same unanswerable share, steps, and tokens.

## 7. Robustness edits (E10, Full)

- decisive_removed: delete the sentence that contains the gold answer. Gold becomes ABSTAIN.
- distractor_added: insert one irrelevant sentence from the pool. Gold stays the same.

## 8. Output files

```
data/
  orgs.jsonl            fact tables, one org per line
  documents/            ORG-0001.md … ORG-0044.md
  items_train.jsonl
  items_val.jsonl
  items_test.jsonl      locked
  items_robust.jsonl
  shortcut_guard.json   question-only classifier accuracy
  TEST_LOCK.txt         sha256 of items_test.jsonl
  MANIFEST.json         counts per split, per answerability, per fact type
```

Each item record has: item_id, org_id, split, fact_type, question_type, template_id, surface, question_text, gold_answer, document_hash, edit_type.

## 9. Acceptance checks (after generation)

1. No organization appears in two splits.
2. Answerable items: the gold answer string appears in the document.
3. Unanswerable items: the field value does not appear in the document.
4. No single answer value exceeds 40% of its fact type.
5. Shortcut guard accuracy ≤ 0.60.
6. Test lock hash matches.
