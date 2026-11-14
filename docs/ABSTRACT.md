# HUMBLE: catch and repair bluffing in fine-tuned RAG models before deployment

**Track 1 (Fine-tuning and Data Curation).** Problem statements #3 (primary), #1, #8. Model: CohereLabs/tiny-aya-global.
Lexsi toolkits: CuratorKIT → AlignTune → AuditKIT.

Companies build retrieval-augmented assistants on their own documents and often fine-tune the model for domain
language or format. Retrieval is imperfect, so the passages often do not contain the answer, and fine-tuning on
answer-only data can teach the model to answer anyway. RAG supplies documents; it does not teach the model to say
"I don't know".

HUMBLE is a local, pre-deployment release gate for fine-tuned RAG models. It builds probes from a company's
documents (four passages per prompt: gold plus distractors, distractors only, answer sentence deleted, misleading
edit; five unanswerable categories: missing fact, out of database, false premise, underspecified, misleading),
verifies them mechanically and with a local sufficient-context judge, audits the candidate and the original model
on the same probes, and applies label-based release rules that do not trust model confidence (bluffing with a
paired, document-clustered CI; over-refusal; accuracy; per-category prudence; format), with thresholds from domain
profiles. A blocked model is repaired from its own mistakes: multisample diagnosis on a separate probe pool, repair
data curated with CuratorKIT (bluffs → ABSTAIN, answerable twins, delete-a-fact and misleading variants, RAFT's
no-gold case flipped to ABSTAIN, 1:1 replay), continued LoRA training with AlignTune, and re-gating. Scoring and
paired significance run through AuditKIT; every decision ships with a reproducible certificate.

On Tiny Aya Global with SQuAD-derived RAG probes (locked test, evaluated once per model), a naive fine-tune raised
accuracy from 52.7% to about 73% but bluffed on every unanswerable question in all three seeds and all five
categories (prudence 40.2% → 0.0%); P(True) confidence did not reveal it. The gate blocked all three. One round of
HUMBLE repair from the blocked adapter passed the gate on every seed, kept the accuracy gain (about 72%), and raised
prudence to about 73% with over-conservativeness near 1%; the repaired model also abstained more on Spanish, Hindi
and Arabic probes and on legal contracts without training on them. We report negative results as pre-registered:
repair data built from the model's own mistakes did not beat a random sample of the same size and format (random
data abstained more; targeted data kept about two more points of accuracy), a fresh honest fine-tune (prevention)
over-refused enough to be blocked on the test set, two-way DPO after repair added prudence but also over-refusal,
and training-free weight scaling could not keep any of the fine-tune without bluffing. Ablations show that replaying
ordinary training data is what prevents over-refusal during repair (without it every seed is blocked), with
answerable twins adding a smaller effect. The practical result is a one-command check that blocks bluffing fine-tunes and a
cheap repair (one round, about 15 GPU-minutes) that restores honest abstention without losing the fine-tune's gains.
