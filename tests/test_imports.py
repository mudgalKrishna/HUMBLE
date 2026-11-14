"""Every module must import (catches syntax errors in GPU-only modules that other tests never touch)."""
import importlib

import pytest

MODULES = ["humble.config", "humble.prompts", "humble.hub", "humble.experiments", "humble.arms", "humble.analysis",
           "humble.cli", "humble.publish", "humble.report", "humble.gate", "humble.repair", "humble.curate",
           "humble.train", "humble.judge", "humble.audit.metrics", "humble.audit.score", "humble.audit.run",
           "humble.audit.generate", "humble.probes.corpus", "humble.probes.squad", "humble.probes.external",
           "humble.probes.docs", "humble.probes.describe"]


@pytest.mark.parametrize("m", MODULES)
def test_import(m):
    importlib.import_module(m)
