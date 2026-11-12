# Run log template

One entry per run. Append to `runs/RUN_LOG.md` (one shared file). Never edit a past entry; add a correction line instead.

```
## <run_id>
- Owner:                 <person A/B/C>
- Date (UTC):            <YYYY-MM-DD HH:MM>
- Model:                 <HF id> @ <revision commit>
- Method:                original | sft | dpo | kto | orpo | grpo | repair_targeted | repair_control
- Seed:                  <int>
- Epoch / steps:         <int> / <int>
- Training items:        <file path> (sha256: <hash>)
- Validation items:      <file path> (sha256: <hash>)
- Test lock ID:          <id>   (test touched? yes/no; must be "no" until final eval)
- Config file:           configs/<name>.yaml (config_hash: <hash>)
- Chat template:         <name or hash>
- Confidence source:     ptrue | verbalized | sampling
- Output path:           runs/<run_id>/predictions.jsonl
- GPU:                   RTX 6000 96GB (or other, state)
- Wall-clock:            <minutes>
- Peak VRAM:             <GB>
- Result summary:        CW=<x>  acc=<x>  sel_acc@80=<x>  abst_unnec=<x>  ECE=<x>
- Status:                complete | failed | partial
- Failure / notes:       <one line>
- Downstream use:        <which experiment (E#) uses this run>
```

## Rules

1. Log the run before it starts (status `running`), then update it at the end.
2. A run without a config file and a dataset hash is not valid. Do not report it.
3. If the test set was touched, the Test lock line must say so and the run cannot be used as a final result.
4. Failed runs stay in the log. Do not delete them.
