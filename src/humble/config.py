"""Config loading. One YAML file (configs/default.yaml) holds paths, model, profiles and settings.

`${VAR:-default}` in string values is expanded from the environment, so the same config runs on a
laptop, a GPU notebook, or CI without edits.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

_ENV_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _expand(v):
    if isinstance(v, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def _find_config(path=None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get("HUMBLE_CONFIG")
    if env:
        return Path(env)
    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG
    packaged = Path(__file__).resolve().parent / "default.yaml"  # wheel installs ship a copy
    return packaged


@lru_cache(maxsize=8)
def _load(path_str: str) -> dict:
    raw = yaml.safe_load(Path(path_str).read_text(encoding="utf-8"))
    return _expand(raw)


def load_config(path=None) -> dict:
    return copy.deepcopy(_load(str(_find_config(path))))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def home(cfg=None) -> Path:
    cfg = cfg or load_config()
    p = Path(cfg["paths"]["home"]).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def path(name: str, cfg=None) -> Path:
    """A named working folder under HUMBLE_HOME (probes, audits, adapters, ...)."""
    cfg = cfg or load_config()
    p = home(cfg) / cfg["paths"][name]
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_source(cfg=None) -> str:
    cfg = cfg or load_config()
    local = cfg["model"].get("local_path")
    if local and Path(local).exists():
        return local
    return cfg["model"]["base_id"]


def profile(name=None, cfg=None) -> dict:
    cfg = cfg or load_config()
    name = name or cfg["gate"]["default_profile"]
    profiles = cfg["gate"]["profiles"]
    if name not in profiles:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(profiles)}")
    return {"name": name, **profiles[name]}


def one_gpu_job_at_a_time(fn):
    """Parallel jobs on one GPU (scripts/*_parallel.sh) OOM a 96 GB card when two trainings or audits overlap
    (each takes 30-45 GB). Serialise GPU work with a file lock in HUMBLE_HOME/state, and hand the cached memory
    back (PyTorch keeps it in this process otherwise) before the next job takes the lock. Not re-entrant: never
    call one wrapped function from inside another."""
    import functools

    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        import fcntl
        import gc
        lock = path("state", kwargs.get("cfg") or load_config()) / "gpu.lock"
        with open(lock, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                return fn(*args, **kwargs)
            finally:
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
    return wrap
