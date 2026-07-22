"""Peaks: density control and gain invariance — the two robustness guarantees."""

from __future__ import annotations

import numpy as np
import pytest

from audio_fingerprint import DEFAULT_CONFIG, synth
from audio_fingerprint.peaks import DEFAULT_PEAK_CONFIG, PeakConfig, pick_peaks


def test_density_near_target_across_signal_levels(small_corpus):
    """Peak density must stay near the target as the input gain varies widely.

    This is the property that makes the retrieval score meaningful: score is
    roughly proportional to the number of query hashes, so if density tracked
    loudness, score would track loudness too and a fixed threshold would be
    meaningless. We fingerprint the same track at gains from -40 to +40 dB and
    require the density to stay within a tolerance band of the target.
    """
    audio = small_corpus.audio[small_corpus.specs[0].track_id]
    target = DEFAULT_PEAK_CONFIG.target_density
    for gain_db in (-40.0, -20.0, 0.0, 20.0, 40.0):
        con = pick_peaks(synth.apply_gain(audio, gain_db), DEFAULT_CONFIG)
        # Quota logic pins density near target; allow a band for edge blocks.
        assert 0.6 * target <= con.density <= 1.5 * target, (
            f"gain {gain_db} dB gave density {con.density:.1f}, target {target}"
        )


def test_peaks_are_gain_invariant(small_corpus):
    """A pure gain change must not move a single peak.

    Working in dB turns a gain into an additive constant, which shifts neither
    local maxima nor a (local mean + k*std) threshold. So the constellation is
    exactly identical, not merely similar. This is asserted exactly.
    """
    audio = small_corpus.audio[small_corpus.specs[1].track_id]
    base = pick_peaks(audio, DEFAULT_CONFIG)
    for gain_db in (-30.0, 12.0, 25.0):
        con = pick_peaks(synth.apply_gain(audio, gain_db), DEFAULT_CONFIG)
        np.testing.assert_array_equal(con.frames, base.frames)
        np.testing.assert_array_equal(con.bins, base.bins)


def test_density_tracks_the_requested_target(small_corpus):
    """Raising target_density raises realised density, monotonically."""
    audio = small_corpus.audio[small_corpus.specs[2].track_id]
    densities = []
    for target in (10.0, 20.0, 40.0):
        con = pick_peaks(audio, DEFAULT_CONFIG, PeakConfig(target_density=target))
        densities.append(con.density)
    assert densities[0] < densities[1] < densities[2]


def test_peaks_avoid_the_sub_bass_floor(small_corpus):
    """No peak below min_band_bin — rumble/DC must be excluded."""
    audio = small_corpus.audio[small_corpus.specs[0].track_id]
    con = pick_peaks(audio, DEFAULT_CONFIG, DEFAULT_PEAK_CONFIG)
    assert con.bins.min() >= DEFAULT_PEAK_CONFIG.min_band_bin


def test_peaks_spread_across_frequency_bands(small_corpus):
    """The banded quota should populate more than one frequency band.

    Without banding a bass-heavy track puts every peak at the bottom of the
    spectrum, and a high-pass-filtered query then matches nothing. We require
    peaks in at least three distinct sixths of the used band.
    """
    audio = small_corpus.audio[small_corpus.specs[3].track_id]
    con = pick_peaks(audio, DEFAULT_CONFIG)
    n_bins = DEFAULT_CONFIG.n_bins
    bands = (con.bins.astype(np.int64) * 6 // n_bins)
    assert np.unique(bands).size >= 3


def test_short_signal_raises(config):
    """Fewer than one frame of audio is a caller bug and must raise."""
    from audio_fingerprint.spectrogram import frame_signal

    with pytest.raises(ValueError):
        frame_signal(np.zeros(10), config.n_fft, config.hop_length)
