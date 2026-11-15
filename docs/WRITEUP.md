# HUMBLE write-up

**Catch and repair bluffing in fine-tuned RAG models before deployment.** Track 1 (Fine-tuning and Data Curation),
Lexsi AI Safety and Alignment Hackathon. Model: CohereLabs/tiny-aya-global. Toolkits: CuratorKIT → AlignTune → AuditKIT.
All tables: [results/v3/RESULTS.md](../results/v3/RESULTS.md) (generated from saved predictions).

## 1. Problem

A company builds a RAG assistant on its documents and fine-tunes the model for its domain. Two things go wrong.
Retrieval is imperfect: the passages often do not contain the answer (wrong document, missing clause, false premise).
And fine-tuning on answer-only data teaches the model to always answer. When the passages do not support an answer,
it invents one.

RAG alone does not fix this. Google's "Sufficient Context" study (ICLR 2025) found that added context can reduce
appropriate abstention; GRAB-RAG (AACL 2026) found small frozen RAG models still answered 41.6% of misleading-context
questions under explicit abstention prompting; and refusal-oriented fixes can cause over-refusal
("Do RAG LMs know when they don't know?", 2025). Cloud vendors ship runtime groundedness checks because RAG systems
still hallucinate. HUMBLE is complementary: a one-time check and repair of the model itself, before release, run locally.

## 2. Method

**Probes (plan §6).** Each prompt has four passages. Answerable probes (C1) put the gold paragraph among same-article
distractors. Unanswerable probes: U1 missing fact (C2 distractors only; C3 the answer sentence deleted), U2 out of
database (the question's entity swapped for one absent from the passages), U3 false premise (SQuAD 2.0's
human-written unanswerable questions), U4 underspecified (the document's entity replaced by its generic kind, plus a
second same-kind document with a same-type candidate answer), U5 misleading evidence (the gold passage edited to state
a wrong, same-type answer; GRAB-RAG's construction). Every probe is checked mechanically (no gold string in an
unanswerable context; answer grounded in the gold passage) and by a local sufficient-context judge (Aya Expanse 8B);
cells are over-generated and the first n passing items kept. Splits are by article; paragraphs used by earlier HUMBLE
iterations are excluded from val/test; the test set is hash-locked and every model is evaluated on it once
(a DONE marker on the Hub enforces this across machines). Each unanswerable probe stores an answerable twin.

**Audit and metrics (§7).** Greedy decoding with the chat template's real stop tokens (fixing the v2 stop-token
mismatch, M7); output `ANSWER: <text>` or `ABSTAIN` (CONF removed, M6). AuditKIT scores answers (SQuAD-normalised F1
≥ 0.8 against any gold) and fingerprints the run. Metrics follow Alignment for Honesty (verified in the paper):
prudence = ABSTAIN on unanswerable; over-conservativeness = ABSTAIN on items the original answered correctly;
honesty = (prudence + 1 − over-conservativeness)/2; plus a cost score with domain weights and U5 "echo" (answers that
repeat the planted wrong answer). 95% CIs from a document-clustered bootstrap (1,000 resamples); paired bootstrap for
differences; AuditKIT's paired significance as a cross-check. Long answers (CUAD) are re-graded by the judge (M9).

**Gate v2 (§8).** G1 bluffing (prudence drop beyond m_bluff with a paired CI excluding 0), G2 over-refusal,
G3 accuracy, G4 per-category prudence, G5 format failure > 5%; thresholds and cost weights from four domain profiles.
Self-controls: original vs itself must PASS, always-abstain must BLOCK on G2/G3, always-answer must BLOCK on G1.

**Repair loop v2 (§9).** Starting from the blocked adapter: audit a fresh, document-disjoint slice of the probe pool
with five samples per item; an item is a bluff only if the model answers on most samples (Multisample). Build the
repair set: bluff → ABSTAIN; its twin → gold; delete-a-fact, misleading and distractors-only (RAFT's no-gold case
flipped to ABSTAIN) variants → ABSTAIN, each verified by the judge; over-refusals → gold; 1:1 replay of ordinary
training data. Curate with CuratorKIT (schema gate, exact dedup, provenance), continue the LoRA adapter with AlignTune
SFT (lr 5e-5, one epoch), re-audit on val, re-gate; up to three rounds.

**Experiments.** E-REPAIR: from the same blocked adapter, A the loop, B a random control matched to A's rounds,
sizes, target ratio, replay and steps, C prevention (fresh honest SFT with 30% unanswerables), D lambda scaling via
AlignTune merge; A, B, C with three seeds. Ablations: A without twins, A without replay, A + two-way DPO, A on two
other blocked seeds. External, evaluation only: XQuAD (es, hi, ar), CUAD contracts, and a product-mode probe set
generated with CuratorKIT from a fictional company's 44 policy documents.

## 3. Results

**H1 supported: naive fine-tuning makes the model bluff.** Three seeds of answer-only SFT raised test accuracy from
52.7% to about 73% and dropped prudence from 40.2% to 0.0%: every unanswerable probe in every category was answered.
On misleading probes the naive model echoed the planted wrong answer in about half the cases.

**H2 supported: confidence did not show it.** P(True)-based confident-wrong stayed low for the naive model while it
bluffed on every unanswerable question; the gate therefore relies on labelled probes.

**H5: the gate behaves.** All naive seeds are blocked (G1 and G4, CIs far from zero); self-controls pass across the
healthcare, finance, legal and support profiles; gates recomputed from saved predictions give identical decisions.

**Repair works from the broken model.** One round of the loop passed the gate on all three seeds, on val and on the
locked test: test accuracy 72.3% (keeping the fine-tune's gain), prudence 72.7%, over-conservativeness 1.7%. Started
from the other two blocked naive seeds it also passed (test prudence 70.9% and 71.8%). Prevention (C, a fresh honest
SFT) reaches 70.4% accuracy and 80.0% prudence but over-refuses more (4.6%): it passes on val and is blocked by G2 on
the locked test in all three seeds.

**H3 not supported: targeting did not beat random.** The matched random control (B) also passed, with higher
prudence (about 80%) and lower cost, while A kept about two points more accuracy. Any curated honest data in the RAG
format with distractors repairs this model; selecting the model's own mistakes did not add measurable value here. A
natural explanation was that the naive model bluffs on essentially every unanswerable item, so "its own mistakes"
are a random sample anyway. A follow-up rules this out ([FIX1_TARGETING.md](../results/v3/FIX1_TARGETING.md)): a
lighter naive model (10 training steps; test prudence 12.6%, so its mistakes are a real, skewed subset) was repaired
by targeted and by matched random data on three seeds, and random data was again significantly better on cost and
prudence in every seed (about 5 points more abstention), while targeted data kept about one point more accuracy.
The likelier reason is coverage: random abstain targets follow the probe design across all five categories, while
the model's own bluffs over-represent the easy U1 category. The same run shows how fast the damage happens:
abstention collapses within 10–20 optimizer steps of answer-only fine-tuning.

**H4 not supported under the strict rule.** A had lower over-conservativeness than C on every seed (by 3.8, 1.9 and
3.2 points), but the paired CI excludes zero only for seed 31. In gate terms the difference matters: A passes the
test gate on every seed and C is blocked on every seed.

**Ablations: replay is the main protection against over-refusal; twins add a smaller one.** A without replay
(twins kept, no ordinary training data) was blocked on G2 in all three seeds, on val and on test; a second round made
it worse, and on test over-conservativeness was 17.9% ± 6.5 with accuracy 58.4%. A without twins passed on val in all
three seeds but was blocked by G2 on the locked test in two of three (over-conservativeness 3.6% ± 1.3 vs 1.7% with
twins). So the 1:1 replay of ordinary answerable data is what keeps the repaired model from over-refusing, and the
answerable twins reduce it further.

**H6 partly supported.** Categories differ strongly; for the original model the hardest category is U4
(underspecified), with U5 (misleading) next; after repair, U3 (false premise), U4 and U5 remain the weakest.

**H7: repair transfers.** On XQuAD the repaired model abstains on about 85% of unanswerable probes (original 55%,
naive 0%); on CUAD contracts on about half (naive 0%), with judge-graded accuracy equal to the naive model's.

**Lambda scaling (D) cannot repair.** Every lambda from 0.05 to 1.0 is blocked; bluffing rises with lambda, so only
lambda = 0 (the original) passes, losing the fine-tune's gains. **Two-way DPO**: the first run was invalid because
AlignTune's DPO `train()` rebuilds the model and silently replaced the repaired adapter with a fresh LoRA (detected
by lora_B norms 0.19 vs 4.49; its scores matched the original model). The wrapper now injects the adapter and checks
the norm. In the valid rerun DPO did continue the adapter (lora_B norm 4.49 → 6.36) and passed on val; on the
locked test it raised prudence over A s31 (75.1% vs 71.8%) at the cost of accuracy (71.0% vs 72.3%) and
over-refusal (3.2% vs 1.3%), and was blocked by G2. Two-way DPO did not improve on SFT repair here. (Its first test
audit loaded the SFT step instead of the DPO adapter; this was caught because its metrics matched A exactly, fixed
by recording the adapter actually evaluated, and the DPO adapter was tested once under its own name.)

## 4. Limitations

* One model (Tiny Aya Global), by design for this release; other Cohere models need their terms accepted.
* The judge filters U4 and U5 heavily (pass rates about 21% and 28%): a sufficiency judge tends to call an edited
  passage "sufficient", so the kept misleading items are those where something looks off even to the judge. U4/U5
  cells are small (28 and 50 test items) and their CIs are wide.
* Judge validation against 100 human labels (E-JUDGE) is prepared (`v3/judge/judge_label_sheet_val.csv`) but needs
  the team's labels.
* XQuAD paragraphs are translations of SQuAD v1.1 dev paragraphs whose English originals may appear in our SQuAD pools.
* Greedy decoding is not bit-exact across batch layouts on GPU: a rerun reproduced 198/200 actions (E-REPRO).
* The product-mode demo uses fictional policy documents; generated QA from a 3B model is noisy.

## 5. Related work

Gekhman et al. (EMNLP 2024) and Kang et al. (NAACL 2025) on fine-tuning and hallucination; OpenAI (2025) on why
models guess; Sufficient Context (ICLR 2025); GRAB-RAG (AACL 2026); "Do RAG LMs know when they don't know?" (2025);
UAEval4RAG (ACL 2025); R-Tuning (NAACL 2024); Alignment for Honesty (NeurIPS 2024); RAFT (2024); GRACE (2026);
Self-RAG (ICLR 2024); Kadavath et al. (2022) for P(True); SQuAD 2.0, XQuAD, CUAD. Full links in docs/PROJECT_v3.md.
