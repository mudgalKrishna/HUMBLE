"""Hugging Face Hub as the communication channel between machines.

Notebooks pull code and data from the HF dataset repo and push status, logs and results back.
The lead machine reads everything from the same repo. Every job is resumable: it skips work whose
output already exists locally or on the Hub.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path

from .config import load_config, path


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def repo(cfg=None):
    """The repo this machine WRITES to. Each notebook node gets its own repo (<repo>-nb<N>), spreading the
    per-repo commit rate limit; reads search every repo (see read_repos)."""
    cfg = cfg or load_config()
    node = os.environ.get("HUMBLE_NODE", "")
    base = cfg["hub"]["repo"]
    if cfg["hub"].get("per_node_repos", True) and node.upper().startswith("NB"):
        return f"{base}-{node.lower()}", cfg["hub"]["repo_type"]
    return base, cfg["hub"]["repo_type"]


def read_repos(cfg=None):
    """Legacy repo first, then the shared v3 repo, then every per-node repo (newer files win on pull)."""
    cfg = cfg or load_config()
    base = cfg["hub"]["repo"]
    out = [r for r in (cfg["hub"].get("legacy_repo"), base) if r]
    if cfg["hub"].get("per_node_repos", True):
        out += [f"{base}-nb{i}" for i in range(1, cfg["hub"].get("n_nodes", 8) + 1)]
    return out


def _push_now(local, remote, rid, rtype, message):
    api = _api()
    if local.is_dir():
        api.upload_folder(repo_id=rid, repo_type=rtype, folder_path=str(local), path_in_repo=remote,
                          commit_message=message or f"push {remote}")
    else:
        api.upload_file(repo_id=rid, repo_type=rtype, path_or_fileobj=str(local), path_in_repo=remote,
                        commit_message=message or f"push {remote}")


def push(local, remote, cfg=None, message=None):
    """Upload a file or folder in one commit. Never blocks longer than hub.push_timeout_s; failures are
    logged to state/push_errors.log and listed in state/pending_pushes.jsonl for `flush_pending`."""
    cfg = cfg or load_config()
    rid, rtype = repo(cfg)
    local = Path(local)
    box = {}

    def _run():
        try:
            _push_now(local, remote, rid, rtype, message)
            box["ok"] = True
        except Exception as e:
            box["err"] = repr(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(cfg["hub"].get("push_timeout_s", 300))
    if box.get("ok"):
        return True
    err = box.get("err", "timeout (probably rate limited; upload continues in background)")
    with open(path("state", cfg) / "push_errors.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {remote}: {err}\n")
    with open(path("state", cfg) / "pending_pushes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"local": str(local), "remote": remote}) + "\n")
    return False


def flush_pending(cfg=None):
    cfg = cfg or load_config()
    p = path("state", cfg) / "pending_pushes.jsonl"
    if not p.exists():
        return 0
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    p.unlink()
    done = 0
    for r in {x["remote"]: x for x in rows}.values():
        done += bool(push(r["local"], r["remote"], cfg))
    return done


def pull(patterns, local_dir, cfg=None):
    """Download matching files: legacy repo first, then the v3 repo (newer files win)."""
    from huggingface_hub import snapshot_download
    cfg = cfg or load_config()
    rtype = cfg["hub"]["repo_type"]
    out = None
    for rid in read_repos(cfg):
        try:
            out = snapshot_download(repo_id=rid, repo_type=rtype, allow_patterns=list(patterns),
                                    local_dir=str(local_dir), token=os.environ.get("HF_TOKEN"), max_workers=4)
        except Exception:
            pass
    return out


def exists_remote(remote_file, cfg=None) -> bool:
    cfg = cfg or load_config()
    for rid in reversed(read_repos(cfg)):
        try:
            if _api().file_exists(rid, remote_file, repo_type=cfg["hub"]["repo_type"]):
                return True
        except Exception:
            pass
    return False


class Job:
    """Status + log for one job; mirrored to the Hub under state/ and logs/."""

    def __init__(self, name, cfg=None, node=None):
        self.name, self.cfg = name, cfg or load_config()
        self.node = node or os.environ.get("HUMBLE_NODE", "local")
        self.state_file = path("state", self.cfg) / f"{name}.json"
        self.log_file = path("logs", self.cfg) / f"{name}.log"
        self._last_push = 0.0

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{self.node}] {msg}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)
        if time.time() - self._last_push > 120:  # at most every 2 minutes
            self.sync()

    def status(self, stage, **extra):
        rec = {"job": self.name, "node": self.node, "stage": stage,
               "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
        self.state_file.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        self.sync()
        return rec

    def sync(self):
        self._last_push = time.time()
        if not self.cfg["hub"].get("sync_state", False):
            return
        push(self.state_file, f"v3/state/{self.name}.json", self.cfg) if self.state_file.exists() else None
        push(self.log_file, f"v3/logs/{self.name}.log", self.cfg) if self.log_file.exists() else None


def run_job(name, fn, cfg=None):
    """Run fn(job) synchronously with status + error capture (used by subprocess launches)."""
    job = Job(name, cfg)
    job.status("running", pid=os.getpid())
    try:
        result = fn(job)
        job.status("done", result=result)
        return result
    except Exception as e:
        job.log("ERROR " + repr(e))
        job.status("error", error=repr(e), trace=traceback.format_exc()[-3000:])
        raise


def run_in_thread(name, fn, cfg=None):
    """Run fn(job) in a background thread so the notebook cell returns at once."""
    job = Job(name, cfg)

    def _wrap():
        job.status("running")
        try:
            result = fn(job)
            job.status("done", result=result)
        except Exception as e:
            job.log("ERROR " + repr(e))
            job.status("error", error=repr(e), trace=traceback.format_exc()[-3000:])

    t = threading.Thread(target=_wrap, name=f"humble-{name}", daemon=False)
    t.start()
    return t
