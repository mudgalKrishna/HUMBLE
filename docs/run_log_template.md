# Run log template

One entry per run. Append to `runs/RUN_LOG.md` (one shared file). Never edit a past entry; add a correction line instead.

```
## <run_id>
- Owner:                 <A / B / C>
- Date (UTC):            <YYYY-MM-DD HH:MM>
- Model:                 <HF id> @ <revision commit>
- Candidate:             original | sft_naive | sft_honest | dpo_naive | dpo_honest |
                         kto_* | orpo_* | grpo_* | repaired_targeted | repaired_control | lambda_scaled
- Method / data recipe:  <method> / <naive | honest | targeted_repair | budget_matched_control>
- Prompt setting:        open_book | closed_book
- Seed:                  <int>
- Epoch / steps:         <int> / <int>
- Training items:        <file path> (sha256: <hash>)
- Validation items:      <file path> (sha256: <hash>)
- Test lock:             <lock id>; test touched? yes / no (must be "no" until final evaluation)
- Config file:           configs/default.yaml (config_hash: <hash>)
- Chat template:         <name or hash>
- ptrue token IDs:       Yes=<id>, No=<id>
- Confidence source:     ptrue | verbalized | sampling
- Samples per item:      10 (or 5, with reason)
- Output folder:         audits/<candidate>/ (predictions.jsonl, metrics.json, gate.json)
- GPU:                   RTX 6000 96GB (or other, state)
- Wall-clock:            <minutes>
- Peak VRAM:             <GB>
- Result summary:        CW=<x>  acc=<x>  SelAcc80=<x>  UnnecAbst=<x>  ECE=<x>  format_fail=<x>
- Gate decision:         PASS | BLOCK (reasons) | n/a
- Status:                running | complete | failed | partial
- Failure / notes:       <one line>
- Used in:               <experiment id, e.g. E2, E5>
```

## Rules

1. Log the run before it starts (status `running`), then update it at the end.
2. A run without a config hash and a dataset hash is not valid. Do not report it.
3. If the test set was touched, the test-lock line must say so, and the run cannot be a final result.
4. Failed runs stay in the log. Do not delete them.
5. Any change to `configs/default.yaml` is a new config hash. Announce it to all owners before the next run.
