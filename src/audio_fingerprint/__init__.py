"""audio-fingerprint — Shazam-style acoustic fingerprinting and retrieval.

Identify a recording from a short, noisy excerpt. The pipeline is:

    audio -> STFT (dB)          spectrogram.py
          -> spectral peaks     peaks.py
          -> paired hashes      hashing.py
          -> inverted index     index.py
          -> offset voting      match.py

**The evaluation corpus is SYNTHETIC** — procedurally generated harmonic tone
sequences, not music. See :mod:`audio_fingerprint.synth` and the README for what
that means for interpreting the reported numbers.
"""

from __future__ import annotations

from .hashing import DEFAULT_HASH_CONFIG, Fingerprint, HashConfig, fingerprint
from .index import FingerprintIndex, IndexStats
from .match import DEFAULT_MATCH_CONFIG, MatchConfig, MatchResult, identify
from .peaks import DEFAULT_PEAK_CONFIG, Constellation, PeakConfig, pick_peaks
from .spectrogram import DEFAULT_CONFIG, SpectrogramConfig

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_HASH_CONFIG",
    "DEFAULT_MATCH_CONFIG",
    "DEFAULT_PEAK_CONFIG",
    "Constellation",
    "Fingerprint",
    "FingerprintIndex",
    "HashConfig",
    "IndexStats",
    "MatchConfig",
    "MatchResult",
    "PeakConfig",
    "SpectrogramConfig",
    "fingerprint",
    "identify",
    "pick_peaks",
    "__version__",
]


def fingerprint_audio(
    audio,  # noqa: ANN001 - numpy array; kept loose to avoid a hard import here
    config: SpectrogramConfig = DEFAULT_CONFIG,
    peak_config: PeakConfig = DEFAULT_PEAK_CONFIG,
    hash_config: HashConfig = DEFAULT_HASH_CONFIG,
) -> Fingerprint:
    """Convenience: audio -> :class:`Fingerprint` in one call.

    The three configs are separate arguments rather than one bundle because they
    are tuned independently and at different times: analysis geometry is fixed
    once for the whole database, peak density is the main robustness knob, and
    hash parameters trade index size against recall.
    """
    return fingerprint(pick_peaks(audio, config, peak_config), hash_config)
