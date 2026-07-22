"""Synthesis and degradation: determinism and that degradations do what they say."""

from __future__ import annotations

import numpy as np

from audio_fingerprint import DEFAULT_CONFIG, synth


def test_corpus_is_deterministic():
    a = synth.make_corpus(4, duration_s=5.0, seed=3)
    b = synth.make_corpus(4, duration_s=5.0, seed=3)
    for spec in a.specs:
        np.testing.assert_array_equal(a.audio[spec.track_id], b.audio[spec.track_id])


def test_impostor_ids_do_not_collide_with_corpus():
    corpus = synth.make_corpus(10, duration_s=3.0, seed=0)
    imp = synth.make_impostor_corpus(10, duration_s=3.0, seed=0)
    corpus_ids = {s.track_id for s in corpus.specs}
    imp_ids = {s.track_id for s in imp.specs}
    assert corpus_ids.isdisjoint(imp_ids)


def test_add_noise_hits_requested_snr():
    """The achieved SNR must match the requested SNR, measured independently."""
    rng = np.random.default_rng(0)
    x = synth.synthesize(
        synth.make_corpus(1, duration_s=6.0, seed=1).specs[0], 6.0
    )
    for snr in (20.0, 10.0, 0.0, -6.0):
        noisy = synth.add_noise(x, snr, rng)
        measured = synth.measured_snr_db(x, noisy)
        assert abs(measured - snr) < 0.5, f"asked {snr}, got {measured}"


def test_gain_scales_amplitude_exactly():
    x = np.ones(100)
    out = synth.apply_gain(x, 20.0)  # +20 dB = x10 amplitude
    np.testing.assert_allclose(out, 10.0, rtol=1e-9)


def test_band_limit_attenuates_out_of_band_energy():
    """A low-pass must remove energy above its cutoff."""
    sr = 11025
    t = np.arange(sr, dtype=np.float64) / sr
    low = np.sin(2 * np.pi * 300 * t)
    high = np.sin(2 * np.pi * 4000 * t)
    x = low + high
    filtered = synth.band_limit(x, sr, high_hz=1000.0)
    # Energy of the filtered signal should be close to the low tone alone.
    assert np.mean(filtered**2) < 0.7 * np.mean(x**2)
    # And correlate strongly with the surviving low tone.
    corr = np.corrcoef(filtered, low)[0, 1]
    assert corr > 0.9


def test_crop_returns_requested_length_and_start():
    rng = np.random.default_rng(2)
    x = np.arange(11025 * 5, dtype=np.float64)
    excerpt, start = synth.crop(x, 11025, 2.0, rng)
    assert excerpt.size == 11025 * 2
    assert 0.0 <= start <= 5.0


def test_speed_change_shifts_frequency():
    """Playing faster raises pitch: the spectral peak moves up."""
    from audio_fingerprint.spectrogram import fft_frequencies, power_spectrogram

    cfg = DEFAULT_CONFIG
    t = np.arange(cfg.sample_rate * 2, dtype=np.float64) / cfg.sample_rate
    x = np.sin(2 * np.pi * 500 * t)
    fast = synth.speed_change(x, 1.06)
    freqs = fft_frequencies(cfg)
    base_bin = int(np.argmax(power_spectrogram(x, cfg).mean(axis=1)))
    fast_bin = int(np.argmax(power_spectrogram(fast, cfg).mean(axis=1)))
    assert freqs[fast_bin] > freqs[base_bin]


def test_time_stretch_preserves_length_relationship():
    """rate > 1 shortens the signal, rate < 1 lengthens it."""
    x = np.random.default_rng(0).standard_normal(11025 * 3)
    shorter = synth.time_stretch(x, 1.2)
    longer = synth.time_stretch(x, 0.8)
    assert shorter.size < x.size < longer.size
