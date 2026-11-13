"""Notebook job runner: one job per notebook, progress saved to Hugging Face, resumable.

Every job writes status to STATE/<job>.json and uploads results to the HF dataset repo.
A job skips work whose output already exists, so a restart resumes from the last saved step.
"""
import json
import os
import threading
import time
from pathlib import Path

REPO = "KM2007/humble-runs"
BASE = Path("/marimo/humble")
DATA = BASE / "data"
RUNS = BASE / "runs"
ADAPTERS = BASE / "adapters"
STATE = BASE / "state"
MODEL = Path("/marimo/models/tiny-aya-global")
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _hf():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def set_status(job, stage, **extra):
    STATE.mkdir(parents=True, exist_ok=True)
    rec = {"job": job, "stage": stage, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    (STATE / f"{job}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def sync_up(local_path, remote_path):
    """Upload a file or folder to the HF progress repo. Failures are logged, not fatal."""
    api = _hf()
    local_path = Path(local_path)
    try:
        if local_path.is_dir():
            api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(local_path),
                              path_in_repo=remote_path, commit_message=f"progress {remote_path}")
        else:
            api.upload_file(repo_id=REPO, repo_type="dataset", path_or_fileobj=str(local_path),
                            path_in_repo=remote_path, commit_message=f"progress {remote_path}")
        return True
    except Exception as e:  # keep running even if upload fails; retried at next save
        (STATE / "upload_errors.log").open("a", encoding="utf-8").write(f"{remote_path}: {e!r}\n")
        return False


def sync_down(remote_path, local_dir):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=[f"{remote_path}/*", f"{remote_path}/**"],
                             local_dir=str(local_dir), token=os.environ["HF_TOKEN"])


def load_items(split):
    rows = []
    for line in (DATA / f"items_{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def log_to(job):
    path = STATE / f"{job}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return _log


def run_in_thread(job, fn, *args, **kwargs):
    """Run a job in a background thread inside the kernel. Errors go to status."""
    def _wrap():
        set_status(job, "running")
        try:
            result = fn(*args, **kwargs)
            set_status(job, "done", result=result)
            sync_up(STATE / f"{job}.json", f"state/{job}.json")
        except Exception as e:
            import traceback
            set_status(job, "error", error=repr(e), trace=traceback.format_exc()[-1500:])
            sync_up(STATE / f"{job}.json", f"state/{job}.json")
    t = threading.Thread(target=_wrap, name=job, daemon=False)
    t.start()
    return t
