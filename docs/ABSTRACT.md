<p align="center"><img src="assets/humble_wordmark.svg" alt="HUMBLE" width="520"></p>

# HUMBLE: catch and repair bluffing in fine-tuned RAG models before deployment

**Track 1 (Fine-tuning and Data Curation).** Problem statements #3 (primary), #1, #8. Models: Tiny Aya Global (main),
Aya Expanse 8B, Tiny Aya Fire and Tiny Aya Earth, all open Cohere Labs models run locally. Lexsi toolkits:
CuratorKIT → AlignTune → AuditKIT.

Company chatbots search their documents and answer from what they find, and companies give them
extra training on their own examples. We found that this training makes the chatbot stop saying "I couldn't find
that" and make up an answer instead, every time, on four out of four Cohere models, while normal tests score it higher.

HUMBLE is a local release check for fine-tuned RAG models. It builds test questions from a company's documents, each
with four passages, including five kinds of question the passages cannot answer (missing fact, unknown entity, false
premise, ambiguous question, misleading passage), checked by rule and by a local judge (Aya Expanse 8B). It runs the
original and the fine-tuned model on the same questions and blocks the fine-tune if it bluffs more, refuses
answerable questions more, loses accuracy, breaks a question type or garbles its format (paired confidence
intervals, domain risk profiles). A blocked model is repaired from its own mistakes: refuse the questions it bluffed
on, answer their answerable twins, mix in normal training data 1:1, and continue the LoRA adapter for one epoch.
Every decision comes with a certificate.

CuratorKIT checked every training set (schema, duplicates, provenance) and builds questions from documents.
AlignTune ran about 70 LoRA training runs through one function. AuditKIT scored every audit (113 in the public
dataset) and cross-checked our statistics. Aya Expanse 8B as the local judge kept the pipeline free of any API.

On Tiny Aya Global (locked test, evaluated once per model), a standard fine-tune raised accuracy from 52.7% to about
73% but answered every unanswerable question, in all three seeds and all five question types. The gate blocked all
three. One round of HUMBLE repair passed on every seed, kept the accuracy (72.3%), cut the bluff rate from 100% to
27% and kept needless refusals near 2%. The repair also transferred to Spanish, Hindi and Arabic questions and to
legal contracts. The same recipe, unchanged, worked on Aya Expanse 8B (bluff rate 100% → 17%), Tiny Aya Fire (→ 26%)
and Tiny Aya Earth (→ 27%), passing every seed each time.

What did not work: repair data built from the model's own mistakes did not beat random data of the same size on
cost, though on Tiny Aya Earth random data and a fresh honest fine-tune over-refused and were blocked on the test set
while HUMBLE repair passed every seed. Training an honest model from scratch, adding DPO after repair, and scaling down
the fine-tune's weights all failed. Mixing in normal training data is what keeps the repair from over-refusing.

The result is a one-command check that blocks bluffing fine-tunes, and a repair that takes about 15 GPU-minutes.
