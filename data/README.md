# Shared HUMBLE v3 data

`humble_v3_shared.tar.gz` holds what every run needs, so the same locked probes are used for every model:

* `probes/squad3/`: training pool (answer-only SFT data for the naive fine-tune and replay).
* `probes/squad3v/`: val, test and repair-pool probes. **The test split is locked**
  (sha256 of `items_test.jsonl` starts `78c6540b174a4626`; see `TEST_LOCK.txt`). Do not edit or rebuild it.
* `audits/squad3v/original/{val,test}/`: Tiny Aya Global's original-model audits. Needed only for new
  experiments on Tiny Aya Global (the gate compares against them; its test is already evaluated and must not be
  rerun). Other base models produce their own `original` audits.

Unpack into your working folder (`HUMBLE_HOME`); the code reads these local files before looking at the Hub:

```bash
mkdir -p work && tar xzf data/humble_v3_shared.tar.gz -C work
# other base models: remove the Tiny Aya audits so you never gate against them
rm -rf work/audits/squad3v/original
```

Licences: derived from SQuAD 2.0 (CC BY-SA 4.0); model outputs from CohereLabs/tiny-aya-global (CC BY-NC 4.0).
