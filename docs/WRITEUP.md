<p align="center"><img src="assets/humble_wordmark.svg" alt="HUMBLE" width="560"></p>

# HUMBLE write-up

> **The short version.** A standard fine-tune made four out of four Cohere models answer 100% of the questions their
> documents could not answer, while accuracy went up, so every normal eval approved them. HUMBLE is the release
> check that blocks such a model, and a 15-minute repair that brings the bluff rate down to 17–27% (lower than the
> original model) while keeping the accuracy gain.

**Catch and repair bluffing in fine-tuned RAG models before deployment.** Track 1 (Fine-tuning and Data Curation),
Lexsi AI Safety and Alignment Hackathon. Models: Tiny Aya Global (main), Aya Expanse 8B, Tiny Aya Fire, Tiny Aya Earth,
all open Cohere Labs models run locally. Lexsi toolkits: CuratorKIT → AlignTune → AuditKIT (how we used each: section 2b).
All tables: [results/v3/RESULTS.md](../results/v3/RESULTS.md) (generated from saved predictions).

## In plain English

Most company chatbots first search the company's documents and then answer from what they found (this is called
RAG). Sometimes the search finds the wrong pages, and the honest reply is "I couldn't find that in our documents".
Companies also give the chatbot extra training on their own examples (fine-tuning). We found that this training
makes the chatbot stop admitting it couldn't find the answer: it makes one up instead, every time, on all four
Cohere models we tested. The usual tests miss it, because they only ask questions that do have answers, and on those
the trained chatbot scores higher. HUMBLE is a safety inspection before release: it quizzes the chatbot with trick
questions whose answers are not in the documents, blocks it if it bluffs, and retrains it for about 15 minutes so it
says "I don't know" again without losing what it learned.

## Who it is for

AI labs that ship fine-tuned or domain models (a release gate in the training pipeline that blocks a bluffing
checkpoint and repairs it with a short run), fine-tuning platforms and model hubs (one check and certificate per
customer model), companies building chatbots on their own documents (test questions from their documents and a
PASS/BLOCK before launch), regulated businesses in health, finance and law (stricter domain profiles and an audit
trail), and AI safety and evaluation teams (honesty metrics that cannot be gamed by refusing everything). All of it
runs locally, so documents never leave the company's machines.

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

## 1b. Choices and why

**Task: RAG question answering where the right move is sometimes "I can't answer from these passages".** This is the
setting companies actually deploy (an assistant over their own documents), and the one where retrieval misses make
abstention necessary. Every prompt carries four passages; the model must answer from them or reply `ABSTAIN`.

**Models: small open Cohere models, run locally.** Tiny Aya Global (3.35B) is the main model because it is the kind a
company can fine-tune and self-host on one GPU, and because small fine-tuned models are where bluffing goes unnoticed.
To test that the recipe is not tuned to one checkpoint (H8) we re-ran it unchanged on Aya Expanse 8B (larger,
different family member, answers in full sentences), Tiny Aya Fire and Tiny Aya Earth. Everything runs locally:
no API calls, the judge included (Aya Expanse 8B).

**Datasets.** SQuAD 2.0 gives real passages, gold answers and human-written unanswerable questions (false premises),
under a licence that allows derived data (CC BY-SA 4.0). From it we build five kinds of unanswerable probe (missing fact,
out of database, false premise, underspecified, misleading evidence), because "unanswerable" in production has
different causes and a model can handle one and fail another. Splits are by article, earlier iterations' paragraphs are
excluded, and the test set is hash-locked and evaluated once per model. Two external sets, never trained on, check
transfer: XQuAD (Spanish, Hindi, Arabic; multilingual is what Tiny Aya is built for) and CUAD (legal contracts; long,
domain-specific answers).

**Metrics: more than accuracy, because accuracy rewards bluffing.** A naive fine-tune *raises* accuracy (52.7% → 73%) while
answering every unanswerable question; accuracy alone would ship it. So every model is scored on:
* **Accuracy** on answerable probes (did the fine-tune's gain survive?);
* **Prudence**: share of unanswerable probes answered with ABSTAIN (does it bluff?);
* **Over-conservativeness**: share of questions the *original* answered correctly that the candidate now refuses (did the
  repair overshoot into refusing?); prudence and over-conservativeness are the pair from *Alignment for Honesty*, so one
  cannot be bought with the other unseen;
* **Honesty** = (prudence + 1 − over-conservativeness)/2, a single summary of that pair;
* **Cost**, a weighted sum of bluffs, needless refusals and wrong answers with weights from a domain profile (a support
  bot and a legal assistant price a bluff differently);
* **Per-category prudence** (U1–U5) and **U5 echo** (repeating a planted wrong answer), because an average can hide a
  category that is completely broken.
All come with document-clustered bootstrap CIs; gate decisions use paired CIs against the original model, so "worse
than before fine-tuning" is a statistical statement, not a threshold on one noisy number.

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

## 2b. How we used Lexsi's toolkits and Cohere's models

| Tool or model | What we used it for | How much it did for us |
|---|---|---|
| **CuratorKIT** | Every training set passes through it before training: schema check, exact duplicate removal and a provenance record. Its QA generator and hallucination gate build test questions from a company's documents (`humble probe`), using a local model. | All SFT and DPO data in the project (naive, prevention, every repair round, the DPO arm) went through it, so each adapter can be traced to the exact data it saw. One thing we learned: its exact dedup removes plain repeats, so upweighting needed each copy to show the passages in a different order. |
| **AlignTune** | All LoRA training through its TRL backend (`create_sft_trainer`), the DPO arm (`create_rl_trainer`) and the weight-scaling arm (`merge_models`). | About 70 training runs on four models went through one training function. It also exposed a real problem: its DPO step rebuilt the model and silently replaced our repaired adapter with a fresh one. We caught it from the adapter weight norms (0.19 instead of 4.49), fixed our wrapper, and re-ran the arm. |
| **AuditKIT** | Scores every answer (normalised F1, abstain, format error, honest), fingerprints each run, and gives a paired significance test (`RunComparison`) that we use to cross-check our own bootstrap. | Every audit folder stores its AuditKIT run record; there are 113 in the public dataset alone, and any gate decision can be recomputed from them on CPU (`scripts/reproduce.py`). |
| **Tiny Aya Global** (3.35B) | Main model for all experiments and ablations. | Small enough to fine-tune and host on one GPU; its multilingual training let us test transfer to Spanish, Hindi and Arabic. |
| **Aya Expanse 8B** | Local judge (does a probe's context contain the answer; are two answers equivalent) and a fourth model for the core recipe. | Kept the whole pipeline local with no API; showed the recipe works on a larger model that answers in full sentences. |
| **Tiny Aya Fire and Earth** | The unchanged recipe on two more models. | Showed the result is not tied to one checkpoint (H8). |

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
A possible (untested) reason is coverage: random abstain targets follow the probe design across all five categories, while
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

**Fix 2: upweighting the weak categories** (Tiny Aya Global, run by a teammate;
[FIX2_CATEGORIES.md](../results/v3/FIX2_CATEGORIES.md), [RESULTS_fix2_categories.md](../results/v3/RESULTS_fix2_categories.md)).
Repeating every U3/U4/U5 repair example twice (Acat; replay resized to keep 1:1) raised squad3v test prudence from
72.6% to 77.3% (U3 53.0 → 59.0, U4 31.0 → 39.3, U5 35.3 → 42.7) and lowered cost from 1.11 to 1.02, with accuracy
about the same (72.5 → 72.1) and unnecessary abstention up by about 0.8 points. On a new, larger U4/U5 test set
(squad3u45: 150 U4 + 150 U5 judged items from paragraphs no squad3/squad3v prompt showed) prudence rose from 33.9%
to 42.2%. The paired gain is significant on seeds 31 and 33 but not on 32, so it is a consistent but
seed-sensitive gain, not a pass under the every-seed rule. Caveats: the blocked model is a retrain of the naive
recipe (`naive_s31b`, because the published adapter was not readable with the teammate's token), so plain repair was
re-run on it (Ab) for a fair comparison; and squad3u45 does not exclude paragraphs from earlier iterations (no access
to that repo), as recorded in its MANIFEST.

**H7: repair transfers.** On XQuAD the repaired model abstains on about 85% of unanswerable probes (original 55%,
naive 0%); on CUAD contracts on about half (naive 0%), with judge-graded accuracy equal to the naive model's.

**H8 supported on three more models. Aya Expanse 8B** (recipe unchanged, run by a teammate;
[RESULTS_aya-expanse-8b.md](../results/v3/RESULTS_aya-expanse-8b.md)). The naive fine-tune removed all abstention
(test prudence 10.5% → 0.0%) and was blocked (G1, G4); A passed on all three seeds (accuracy 90.4%, prudence 82.8%,
over-conservativeness 0.4%); B passed with more abstention (88.0%), the same pattern as on Tiny Aya; C passed here
(prudence 82.0%, over-conservativeness 2.4%). Aya answers in full sentences, which token F1 scores as wrong
(the original looked like 23% accurate and G2 had only 69 known items), so every answer F1 rejected was re-graded
by the judge and all gates recomputed from saved predictions (original 84.7%, 254 known items). Two lessons: the
original Aya model already abstains rarely (10.5%), so HUMBLE repair leaves it far more honest than before
fine-tuning; and a release gate must grade answers in the model's own style, or it will under-count over-refusal.

*Tiny Aya Fire* (run by a teammate; [RESULTS_tiny-aya-fire.md](../results/v3/RESULTS_tiny-aya-fire.md)): naive
blocked (test prudence 36.0% → 0.0%, G1 and G4); A passed on all three seeds (accuracy 71.1%, prudence 74.2%,
over-conservativeness 2.2%); B passed with more abstention (81.5%) and beat A on all three seeds, so H3 is not
supported on Fire either. Prevention (C) was not run on Fire.

*Tiny Aya Earth* ([RESULTS_tiny-aya-earth.md](../results/v3/RESULTS_tiny-aya-earth.md)): all three
naive seeds blocked (test prudence 22.0% → 0.0%, G1 and G4); A passed on every seed (accuracy 71.9%, prudence
73.4%, over-conservativeness 1.8%); C reached 81.5% prudence with 4.8% over-conservativeness; repair without replay
was blocked by G2 on every seed, as on Tiny Aya Global. On XQuAD the repaired Earth model abstained on 81.4% of
unanswerable probes (original 39.3%, naive 0%). Locked-test certificates: A passed on all three seeds; the random
control B (prudence 79.6%, over-conservativeness 3.2%) was blocked by G2 on one seed of three; prevention C was
blocked by G2 on all three. On Earth the HUMBLE repair was the only arm to pass on every seed, although B and C
abstained more, which is the over-refusal trade-off the gate is built to catch.

**Lambda scaling (D) cannot repair.** Every lambda from 0.05 to 1.0 is blocked; bluffing rises with lambda, so only
lambda = 0 (the original) passes, losing the fine-tune's gains. **Two-way DPO**: the first run was invalid because
AlignTune's DPO `train()` rebuilds the model and silently replaced the repaired adapter with a fresh LoRA (detected
by lora_B norms 0.19 vs 4.49; its scores matched the original model). The wrapper now injects the adapter and checks
the norm. In the valid rerun DPO did continue the adapter (lora_B norm 4.49 → 6.36) and passed on val; on the
locked test it raised prudence over A s31 (75.1% vs 71.8%) at the cost of accuracy (71.0% vs 72.3%) and
over-refusal (3.2% vs 1.3%), and was blocked by G2. Two-way DPO did not improve on SFT repair here. (Its first test
audit loaded the SFT step instead of the DPO adapter; this was caught because its metrics matched A exactly, fixed
by recording the adapter actually evaluated, and the DPO adapter was tested once under its own name.)

## 3b. Before and after: what changes for a user

Locked test, SQuAD-derived RAG probes. "Bluff rate" = 100 − prudence (answers given when the passages do not contain
the answer). Over-refusal = over-conservativeness.

| Model | Version | Accuracy | Bluff rate | Over-refusal | Gate |
|---|---|---|---|---|---|
| Tiny Aya Global | original | 52.7 | 59.8 | – | reference |
| | naive fine-tune (3 seeds) | ≈ 73.3 | **100.0** | 0.0 | BLOCK |
| | **HUMBLE-repaired (3 seeds)** | **72.3** | **27.3** | 1.7 | **PASS** |
| Aya Expanse 8B | original | 84.7 | 89.5 | – | reference |
| | naive fine-tune | 90.0 | **100.0** | 0.0 | BLOCK |
| | **HUMBLE-repaired (3 seeds)** | **90.4** | **17.2** | 0.4 | **PASS** |
| Tiny Aya Fire | original | 60.0 | 64.0 | – | reference |
| | naive fine-tune | 72.3 | **100.0** | 0.0 | BLOCK |
| | **HUMBLE-repaired (3 seeds)** | **71.1** | **25.8** | 2.2 | **PASS** |
| Tiny Aya Earth | original | 55.7 | 78.0 | – | reference |
| | naive fine-tune (3 seeds) | ≈ 73.6 | **100.0** | 0.0 | BLOCK |
| | **HUMBLE-repaired (3 seeds)** | **71.9** | **26.6** | 1.8 | **PASS** |

(Aya Expanse accuracy and over-refusal are judge-graded; see its results file.)

In use this means: the company keeps the fine-tune's accuracy gain (within about 1–2 points), and when retrieval
misses, the assistant says it cannot answer instead of inventing an answer in roughly three out of four cases
(Tiny Aya) to five out of six (Aya Expanse), where the shipped fine-tune invented one every time and even the original
model did in 60–90% of cases. Needless refusals stay at 0.4–2.2%. The repaired model also abstains more on
Spanish, Hindi and Arabic passages and on legal contracts it was never trained on. The check, the repair and the
certificate take about 0.4 GPU-hours on one GPU; the cost comparison with closed APIs is in [ROI.md](ROI.md).

## 4. Limitations

* Four models: Tiny Aya Global (all experiments), Aya Expanse 8B (core recipe only; judge-graded accuracy, and
  the judge is the same model family), Tiny Aya Fire (no prevention arm) and Tiny Aya Earth (full recipe, three seeds).
* The judge filters U4 and U5 heavily (pass rates about 21% and 28%): a sufficiency judge tends to call an edited
  passage "sufficient", so the kept misleading items are those where something looks off even to the judge. U4/U5
  cells are small (28 and 50 test items) and their CIs are wide.
* Judge validation against 100 human labels (E-JUDGE) is prepared ([data/judge/judge_label_sheet_val.csv](../data/judge/judge_label_sheet_val.csv): fill
  `human_sufficient` with yes/no) but still needs the team's labels (Fix 3, open).
* XQuAD paragraphs are translations of SQuAD v1.1 dev paragraphs whose English originals may appear in our SQuAD pools.
* Greedy decoding is not bit-exact across batch layouts on GPU: a rerun reproduced 198/200 actions (E-REPRO).
* The product-mode demo uses fictional policy documents; generated QA from a 3B model is noisy.

## 5. Related work

Gekhman et al. (EMNLP 2024) and Kang et al. (NAACL 2025) on fine-tuning and hallucination; OpenAI (2025) on why
models guess; Sufficient Context (ICLR 2025); GRAB-RAG (AACL 2026); "Do RAG LMs know when they don't know?" (2025);
UAEval4RAG (ACL 2025); R-Tuning (NAACL 2024); Alignment for Honesty (NeurIPS 2024); RAFT (2024); GRACE (2026);
Self-RAG (ICLR 2024); Kadavath et al. (2022) for P(True); SQuAD 2.0, XQuAD, CUAD. Full links in docs/PROJECT_v3.md.
