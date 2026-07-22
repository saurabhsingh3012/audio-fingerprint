"""Shared fixtures and import-path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_fingerprint import DEFAULT_CONFIG, synth  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def small_corpus():
    """A tiny SYNTHETIC corpus, generated once for the whole session."""
    return synth.make_corpus(6, duration_s=8.0, seed=42)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
