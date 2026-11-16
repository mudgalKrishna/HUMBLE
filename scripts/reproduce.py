"""Reproduce HUMBLE's headline result on a laptop: CPU only, no GPU, no token, about two minutes.

    pip install huggingface_hub pyyaml
    python scripts/reproduce.py

1. Downloads the saved locked-test predictions of three Tiny Aya Global models from the public dataset
   KM2007/humble-v3-probes-and-audits: the original model, the naive fine-tune and the HUMBLE-repaired model.
2. Re-runs the release gate on them: the naive fine-tune must be BLOCKED, the repaired one must PASS.
3. Regenerates results/v3/RESULTS.md from the same public data (it should match the committed file).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "repro"
DATASET = "KM2007/humble-v3-probes-and-audits"
MODELS = ["original", "naive_s31", "repA_s31_r1"]


def main():
    from huggingface_hub import snapshot_download

    print(f"1/3 downloading saved predictions from {DATASET} ...")
    snapshot_download(DATASET, repo_type="dataset", local_dir=str(WORK),
                      allow_patterns=[f"v3/audits/squad3v/{m}/test/*" for m in MODELS])
    audits = WORK / "v3" / "audits" / "squad3v"

    sys.path.insert(0, str(ROOT / "src"))
    from humble.cli import main as humble

    print("\n2/3 release gate on the locked test (profile: support)")
    results = {}
    for cand in ("naive_s31", "repA_s31_r1"):
        print(f"\n$ humble gate original {cand}")
        results[cand] = humble(["gate", str(audits / "original" / "test"), str(audits / cand / "test"),
                                "--profile", "support", "--out", str(WORK / f"gate_{cand}.json")])

    print("\n3/3 regenerating results/v3/RESULTS.md from the public data ...")
    env = dict(os.environ, HUMBLE_HF_REPO=DATASET)
    env.pop("HUMBLE_MODEL_ID", None)
    subprocess.run([sys.executable, str(ROOT / "results" / "v3" / "make_report.py"), str(WORK / "report")],
                   env=env, check=True)

    ok = results["naive_s31"] == 1 and results["repA_s31_r1"] == 0
    print("\n" + ("REPRODUCED: naive fine-tune BLOCKED, HUMBLE-repaired PASSED." if ok
                  else "MISMATCH: gate decisions differ from the published ones."))
    print("Compare the regenerated table with git:  git diff results/v3/RESULTS.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
