"""Spectrogram primitives: framing, windowing, dB scaling, and mel mapping."""

from __future__ import annotations

import numpy as np
import pytest

from audio_fingerprint.spectrogram import (
    SpectrogramConfig,
    amplitude_to_db,
    fft_frequencies,
    frame_signal,
    hann_window,
    hz_to_mel,
    mel_to_hz,
    power_spectrogram,
    stft,
)


def test_frame_signal_geometry():
    x = np.arange(100, dtype=np.float64)
    frames = frame_signal(x, frame_length=10, hop_length=5)
    assert frames.shape == (19, 10)
    np.testing.assert_array_equal(frames[0], np.arange(10))
    np.testing.assert_array_equal(frames[1], np.arange(5, 15))


def test_hann_window_endpoints_periodic():
    w = hann_window(8, periodic=True)
    assert w[0] == pytest.approx(0.0)
    assert w.max() <= 1.0
    assert w.size == 8


def test_stft_recovers_a_pure_tone():
    """A sinusoid at a bin centre concentrates energy in that bin."""
    cfg = SpectrogramConfig(sample_rate=8000, n_fft=1024, hop_length=256)
    freqs = fft_frequencies(cfg)
    target_bin = 64
    f = freqs[target_bin]
    t = np.arange(8000, dtype=np.float64) / cfg.sample_rate
    x = np.sin(2 * np.pi * f * t)
    power = power_spectrogram(x, cfg)
    peak_bin = int(np.argmax(power.mean(axis=1)))
    assert abs(peak_bin - target_bin) <= 1


def test_db_scaling_is_gain_additive():
    """A gain multiplies power; in dB that is an additive constant everywhere."""
    rng = np.random.default_rng(0)
    power = np.abs(rng.standard_normal((50, 20))) + 1e-6
    db = amplitude_to_db(power)
    db_scaled = amplitude_to_db(power * 100.0)  # +20 dB in power terms
    diff = db_scaled - db
    np.testing.assert_allclose(diff, 20.0, atol=1e-9)


def test_mel_round_trip():
    hz = np.array([0.0, 100.0, 440.0, 1000.0, 4000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(hz)), hz, rtol=1e-9)


def test_stft_shape_matches_config():
    cfg = SpectrogramConfig(sample_rate=11025, n_fft=512, hop_length=128)
    x = np.random.default_rng(1).standard_normal(11025)
    spec = stft(x, cfg)
    assert spec.shape[0] == cfg.n_bins
    assert spec.dtype == np.complex128


def test_config_derived_quantities():
    cfg = SpectrogramConfig(sample_rate=11025, n_fft=1024, hop_length=256)
    assert cfg.n_bins == 513
    assert cfg.frames_per_second == pytest.approx(11025 / 256)
    assert cfg.bin_hz == pytest.approx(11025 / 1024)
    assert cfg.frames_to_seconds(43) == pytest.approx(43 * 256 / 11025)
