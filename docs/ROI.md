# ROI: HUMBLE on a self-hosted Tiny Aya

Every input below is either **measured** in this project or an **assumption** stated explicitly so a company can
substitute its own numbers. No price is presented as a quote.

## Measured (one RTX PRO 6000 Blackwell, 96 GB, Tiny Aya Global in bf16)

| Step | Size | Wall time |
|---|---|---|
| Audit (greedy, AuditKIT scoring, bootstrap) | ~1,000 four-passage probes | ~2–3 min |
| Naive or honest LoRA SFT | 2,000–2,860 examples, 2 epochs | ~14–19 min |
| One repair round (5-sample diagnosis on ~760 probes, curate, 1-epoch SFT on ~3,200 examples, val audit, gate) | – | ~15 min |
| Gate, certificate | CPU | seconds |

A full HUMBLE pass on a new fine-tune (audit original + candidate, gate, one repair round, re-gate, certificate)
is roughly **0.4 GPU-hours**.

## Cost of one HUMBLE pass

`cost_pass = 0.4 h × G`, where **G** is the hourly cost of one comparable GPU (assumption; e.g. G = $2/h gives $0.80).

## Value: bluffs avoided per month

`bluffs_avoided = Q × u × (b_before − b_after)`

* **Q** queries per month (assumption, e.g. 50,000)
* **u** share of queries whose retrieved passages do not contain the answer (assumption, e.g. 20%)
* **b_before** bluff rate of the shipped fine-tune: measured 100% (naive SFT, test)
* **b_after** bluff rate after repair: measured about 27–28% (arm A, test)

Example: 50,000 × 0.20 × 0.72 ≈ **7,200 unsupported answers avoided per month**.
With **c** = cost of one unsupported answer reaching a user (assumption: a support re-contact might be $5; a
regulated-domain error far more), monthly value = 7,200 × c (e.g. $36,000 at c = $5) against a one-time cost of
well under $10 of GPU time per fine-tune.

## Self-hosted Tiny Aya vs a closed API model

`monthly_api = Q × T × P_api / 1e6` vs `monthly_self = H × G`

* **T** tokens per query including four retrieved passages: measured prompt ≈ 700–900 tokens + ≤ 48 output tokens
* **P_api** blended price per million tokens of the closed model (assumption; fill in the vendor's current price)
* **H** GPU-hours per month needed to serve Q queries (assumption; batched greedy generation in this project handled
  ~1,000 four-passage prompts in under a minute on one GPU)

Self-hosting also keeps company documents on company hardware, and HUMBLE's check, repair and certificate run on the
same machine; a closed API cannot be repaired by the customer, only guarded at runtime.
