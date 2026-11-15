# ROI: a HUMBLE-repaired Tiny Aya, self-hosted, vs closed-source APIs

Every input is either **measured** in this project or a **stated assumption** that a company can replace with its
own number. Prices are list prices as of November 2026 (sources at the end); they change often, so re-check them.

## 1. Measured in this project (one NVIDIA RTX PRO 6000 Blackwell, 96 GB, Tiny Aya in bf16)

| Quantity | Value | How measured |
|---|---|---|
| Throughput, four-passage RAG prompts | **≈ 3.4 prompts/s** (120 prompts in ≈ 35.5 s) | audit logs, plain `transformers` batched greedy generation (no vLLM, no quantisation) |
| Prompt length | ≈ 700–900 tokens (four passages + question) | tokenizer counts on the probes |
| Output length | ≤ 48 tokens (`ANSWER: <short answer>` or `ABSTAIN`; typically ≈ 10–20) | generation cap / predictions |
| Token throughput | ≈ 2,900 input tok/s + ≈ 50 output tok/s per GPU | 3.4 × 850 and 3.4 × 16 |
| One HUMBLE pass on a new fine-tune (audit original + candidate, gate, one repair round, re-gate, certificate) | **≈ 0.4 GPU-hours** | wall time of the runs; a full arm A on Tiny Aya Earth took ≈ 15 min |

The throughput figure is conservative: a serving stack such as vLLM, or a smaller batch layout, would change it;
we did not measure those and do not claim a number for them.

## 2. Scenario

* **Q = 1,000,000 RAG queries per month** (≈ 33,000 per day: a mid-sized customer-support or internal-docs assistant).
* Per query: **850 input tokens** (four retrieved passages + question) and **16 output tokens** (short answer or ABSTAIN).
  Monthly volume: 850 M input tokens, 16 M output tokens.
* GPU price **G**: $3.27 per GPU-hour (mean on-demand price of an RTX PRO 6000 across 9 providers, Oct 2026); low-cost
  providers list ≈ $0.95/h.

## 3. Monthly cost at 1 M queries

`API cost = 850 × P_in + 16 × P_out` (prices in $ per million tokens).

| Option | $ / 1M input | $ / 1M output | Monthly cost | Cost per 1,000 queries |
|---|---|---|---|---|
| Claude Opus 5.5 (API) | 4.00 | 20.00 | **$3,720** | $3.72 |
| Gemini 3.1 Pro (API, ≤ 200K context) | 2.00 | 12.00 | **$1,892** | $1.89 |
| Claude Sonnet 5.5 (API) | 2.00 | 10.00 | **$1,860** | $1.86 |
| Claude Haiku 5.5 (API) | 0.10 | 0.50 | **$93** | $0.09 |
| **Tiny Aya + HUMBLE, self-hosted, billed per GPU-second** (1 M / 3.4 per s = 82 GPU-h × $3.27) | – | – | **$269** | $0.27 |
| **Tiny Aya + HUMBLE, one dedicated GPU, always on** (730 h × $3.27; capacity ≈ 8.9 M queries/month) | – | – | **$2,387** | $2.39 (falls with volume) |
| same, low-cost provider (730 h × $0.95) | – | – | **$694** | $0.69 |

Plus a one-time **≈ $1.30** of GPU time (0.4 h × $3.27) for each HUMBLE check-and-repair of a new fine-tune.

## 4. Break-even and what the table says honestly

* **Against frontier and mid-tier APIs (Opus, Sonnet, Gemini Pro), self-hosting wins at moderate volume.** Billed per
  GPU-second it is 7–14× cheaper at 1 M queries. A dedicated always-on GPU at $3.27/h breaks even with Sonnet 5.5
  at ≈ 1.3 M queries/month and with Opus 5.5 at ≈ 0.64 M; at the low-cost $0.95/h rate, break-even with Sonnet is
  ≈ 0.37 M queries/month. Above break-even the dedicated GPU's cost stays flat up to its ≈ 8.9 M-query capacity.
* **Against the cheapest closed tier (Claude Haiku 5.5 at $0.10/$0.50), self-hosting is not cheaper** at any volume
  one GPU can serve: $93/month vs $269 even with per-second billing. A cost-only case for self-hosting a 3B model
  against the cheapest API does not hold, and we do not claim it.
* We did **not** measure closed models on our probes, so this is a cost comparison, not a quality comparison.

## 5. What the money buys that a price table does not show

* **The model can be checked and repaired by its owner.** HUMBLE's gate found that a routine fine-tune turned a model
  that abstained on 10–40% of unanswerable questions into one that bluffs on 100% of them (all four models tested), and
  one ≈ 15-minute repair raised prudence to ≈ 73–83% with ≈ 0.4–2% over-refusal and the fine-tune's accuracy kept.
  A closed API can only be guarded at runtime; its weights cannot be audited, repaired or certified by the customer.
* **Data stays on company hardware.** Retrieved company documents never leave the machine; the check, the repair and
  the certificate run on the same GPU.
* **Bluffs avoided.** With u = 20% of queries whose retrieved passages lack the answer (assumption), 1 M queries/month
  and the measured bluff rates (100% naive → ≈ 27% after repair on the locked test), repair avoids
  1,000,000 × 0.20 × 0.73 ≈ **146,000 unsupported answers per month**. At an assumed $5 per unsupported answer that
  reaches a user (a support re-contact; far more in regulated domains) that is ≈ $730,000/month of avoided cost for
  ≈ $1.30 of GPU time per fine-tune.

## 6. Formulas, to plug in your own numbers

```
api_monthly      = Q × (T_in × P_in + T_out × P_out) / 1e6
self_per_second  = Q / throughput_per_s / 3600 × G
self_dedicated   = 730 × G × ceil(Q / (throughput_per_s × 3600 × 730))
humble_one_time  = 0.4 × G
bluffs_avoided   = Q × u × (b_before − b_after)
```

## Sources (prices, November 2026)

* Claude API list prices: Anthropic pricing (Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 per MTok,
  Haiku prompts ≤ 100K tokens).
* Gemini 3.1 Pro: $2/$12 per MTok (≤ 200K context), as listed by
  [CloudZero](https://www.cloudzero.com/blog/gemini-pricing/) and [Finout](https://www.finout.io/blog/gemini-pricing-in-2026);
  confirm on ai.google.dev.
* RTX PRO 6000 Blackwell rental: mean $3.27/GPU-h across 9 providers (Oct 2026,
  [Thunder Compute](https://www.thundercompute.com/blog/nvidia-rtx-pro-6000-pricing)); from $0.95/GPU-h at Nebius
  ([computeprices](https://computeprices.com/providers/nebius/gpus/rtx-pro-6000)).
