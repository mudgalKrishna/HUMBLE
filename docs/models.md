# HUMBLE model roster (6 models)

Status: pre-training. Each row must be confirmed by the owner on the model card before download. Licenses and sizes below come from web search and are not yet checked against each card.

Cohere Labs models get priority. Three are Cohere Labs models (all CC BY-NC 4.0, gated, noncommercial). Three are Apache 2.0 from other labs.

| # | Model (HF ID) | Lab | Size | License | Gated | Role | Owner |
|---|---|---|---|---|---|---|---|
| 1 | CohereLabs/tiny-aya-global | Cohere Labs | 3.35B (confirmed on card summary) | CC BY-NC 4.0 + Cohere AUP | Yes | PRIMARY: demo model, main grid | Person A |
| 2 | CohereLabs/tiny-aya-fire | Cohere Labs | not confirmed (verify on card) | CC BY-NC 4.0 + Cohere AUP | Yes | Regional variant (South Asian languages); same family as #1 | Person A |
| 3 | CohereLabs/aya-expanse-8b | Cohere Labs | 8B | CC BY-NC 4.0 + Cohere AUP | Yes | Scale check (8B) | Person B |
| 4 | CohereLabs/c4ai-command-r7b-12-2024 | Cohere Labs | ~7B (verify) | CC BY-NC 4.0 | Yes | Second Cohere family, scale check | Person B |
| 5 | Qwen/Qwen3-4B | Alibaba Qwen | ~4B (verify) | Apache 2.0 | No | Non-Cohere family, small | Person C |
| 6 | HuggingFaceTB/SmolLM3-3B | Hugging Face | 3B | Apache 2.0 | No | Non-Cohere family, small | Person C |

Reserve (use only if a primary model fails the pilot): mistralai/Ministral-3-3B-Instruct-2512 (Apache 2.0, Dec 2025); allenai/OLMo-2-0425-1B (Apache 2.0, card says "research and educational use"); Qwen/Qwen3-1.7B (Apache 2.0).

## Checks every model must pass in the pilot (E0) before any training

1. Logit access: can we read token probabilities for "Yes" and "No" (needed for P(True)).
2. Format compliance: at least 95% of outputs match `ANSWER: ... | CONF: ...` or `ABSTAIN`.
3. Parser agreement: at least 95% against 50 hand-labeled items.
4. Chat template recorded: the exact template string is saved in the run config.
5. License and gate: the owner has accepted the terms and saved the card's license text in the model card folder.

A model that fails a check is dropped and replaced with a reserve model. Log the reason in `docs/models.md`.

## Publishing constraints

- All Cohere Labs models are noncommercial. Adapters derived from them must carry the CC BY-NC 4.0 license and the Cohere AUP on their model cards.
- Apache 2.0 adapters carry Apache 2.0 and the base model's license notice.
- The hackathon collections must be public and ungated. Confirm with the organizers that publishing gated-base adapters is allowed. If not, publish the code and configs only, plus predictions.

## Verification status (fill in on the day)

| Model | Card read | License confirmed | Size confirmed | Logits checked | Pilot E0 |
|---|---|---|---|---|---|
| tiny-aya-global | | | | | |
| tiny-aya-fire | | | | | |
| aya-expanse-8b | | | | | |
| c4ai-command-r7b-12-2024 | | | | | |
| Qwen3-4B | | | | | |
| SmolLM3-3B | | | | | |
