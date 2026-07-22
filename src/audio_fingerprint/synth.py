"""SYNTHETIC reference corpus and query degradation.

    ***********************************************************************
    * EVERY AUDIO SIGNAL IN THIS PROJECT IS SYNTHETIC. There is no music,  *
    * no recordings, no public dataset. The "tracks" are procedurally      *
    * generated harmonic tone sequences.                                   *
    ***********************************************************************

Why synthetic, and what it costs
--------------------------------
The project is built with no network access, so no audio corpus could be
downloaded. Synthesis was the only way to get an evaluation with *ground truth*
at all — and a real evaluation on synthetic audio is worth far more than an
unevaluated system on imagined audio.

But be honest about what this buys and what it does not. **Retrieval on this
corpus is easier than retrieval on real music**, for reasons that are structural,
not incidental:

* **Stationarity.** Synthetic notes have clean harmonic stacks with stable
  partials. Real instruments have vibrato, inharmonicity, attack transients and
  formant motion, all of which jitter peak locations between the reference
  encoding and the query encoding.
* **Separation.** Each track here is generated from an independently sampled
  timbre and root, so tracks are spectrally distinctive by construction. A real
  corpus contains covers, remasters, live versions, alternate mixes and — worst
  of all — a hundred songs using the same four chords, the same drum samples and
  the same mastering chain.
* **No production chain.** Real queries pass through lossy codecs, dynamic-range
  compression, room reverb, phone microphones with sharp and irregular frequency
  responses, and (for a needle-drop) surface noise, clicks and wow/flutter. Only
  a few of those are modelled here.
* **Scale.** Sixty tracks is not sixty million. False-accept pressure grows with
  corpus size; a hash that is specific enough at N=60 may not be at N=10^7.

So the numbers this project reports are **upper bounds** on real-world behaviour,
and they are reported to characterise *how the algorithm degrades along each
axis*, not to claim a production-quality identification rate. Nothing here should
be read as comparable to a commercial system's real-world accuracy.

The degradation model
---------------------
Query degradations are the part of this module that generalises best, because
they are the standard robustness axes for audio retrieval and are implemented
against real definitions (measured SNR, real Butterworth filters, real
resampling). They are:

* additive white Gaussian noise at a **measured** SNR,
* band-limiting with a Butterworth band-pass / high-pass / low-pass,
* time cropping to a short excerpt from a random position,
* gain change,
* speed change by resampling (models turntable speed error — pitch *and* tempo
  shift together, which is what actually happens when a platter runs fast),
* tempo-only time-stretch by overlap-add (models a broadcast time-compressor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, resample_poly, sosfiltfilt

from .spectrogram import DEFAULT_CONFIG, SpectrogramConfig, hann_window

__all__ = [
    "TrackSpec",
    "add_noise",
    "apply_gain",
    "band_limit",
    "crop",
    "make_corpus",
    "measured_snr_db",
    "speed_change",
    "synthesize",
    "time_stretch",
]


@dataclass(frozen=True)
class TrackSpec:
    """Parameters that define one SYNTHETIC "track".

    Attributes:
        track_id: Stable identifier.
        name: Display name, e.g. ``"SYNTH-017"``.
        root_hz: Fundamental of the lowest scale degree.
        scale: Semitone offsets forming the track's pitch material.
        harmonic_gains: Per-harmonic amplitude multipliers. This is the track's
            "timbre" and is the main thing making tracks distinguishable — two
            tracks on the same root with different harmonic gains have visibly
            different constellations.
        notes_per_second: Event rate.
        pad_partials_hz: Sustained background partials, giving the spectrogram
            persistent horizontal lines. Included because real recordings almost
            always have *some* stationary content, and a fingerprinter that only
            ever sees transients is being tested on an unrealistically easy
            signal.
        click_level: Amplitude of broadband transients at note onsets. Adds
            vertical spectrogram structure, which is the hardest kind for a
            peak picker to handle consistently.
    """

    track_id: int
    name: str
    root_hz: float
    scale: tuple[int, ...]
    harmonic_gains: tuple[float, ...]
    notes_per_second: float
    pad_partials_hz: tuple[float, ...] = ()
    click_level: float = 0.0
    seed: int = 0


def _adsr(n: int, sr: int, attack: float = 0.01, release: float = 0.12) -> NDArray[np.float64]:
    """Simple attack/release envelope; avoids clicks at note boundaries."""
    env = np.ones(n, dtype=np.float64)
    a = min(int(attack * sr), n // 2)
    r = min(int(release * sr), n - a)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[n - r :] *= np.linspace(1.0, 0.0, r) ** 2
    return env


def make_track_spec(track_id: int, rng: np.random.Generator) -> TrackSpec:
    """Sample one SYNTHETIC track specification."""
    root_semitone = int(rng.integers(0, 25))
    root_hz = 55.0 * 2.0 ** (root_semitone / 12.0)
    scale_pool = [
        (0, 2, 4, 5, 7, 9, 11),  # major
        (0, 2, 3, 5, 7, 8, 10),  # natural minor
        (0, 3, 5, 7, 10),  # minor pentatonic
        (0, 1, 5, 7, 8),  # phrygian-ish
        (0, 2, 4, 6, 8, 10),  # whole tone
    ]
    scale = scale_pool[int(rng.integers(0, len(scale_pool)))]
    n_harm = int(rng.integers(6, 15))
    alpha = float(rng.uniform(0.6, 1.8))
    base = np.arange(1, n_harm + 1, dtype=np.float64) ** (-alpha)
    jitter = rng.uniform(0.25, 1.0, size=n_harm)
    gains = tuple(float(g) for g in base * jitter)
    n_pad = int(rng.integers(1, 4))
    pads = tuple(float(f) for f in rng.uniform(300.0, 3500.0, size=n_pad))
    return TrackSpec(
        track_id=track_id,
        name=f"SYNTH-{track_id:03d}",
        root_hz=root_hz,
        scale=scale,
        harmonic_gains=gains,
        notes_per_second=float(rng.uniform(2.0, 6.0)),
        pad_partials_hz=pads,
        click_level=float(rng.uniform(0.0, 0.05)),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


def synthesize(
    spec: TrackSpec,
    duration_s: float,
    sample_rate: int = DEFAULT_CONFIG.sample_rate,
) -> NDArray[np.float64]:
    """Render a SYNTHETIC track from its spec.

    Args:
        spec: Track parameters.
        duration_s: Length in seconds.
        sample_rate: Output sample rate.

    Returns:
        Mono float64 signal normalised to RMS 0.1, so that the SNR helper below
        has a well-defined reference level and so all tracks sit at a comparable
        loudness (a real corpus would not — see the ROADMAP).

    Deterministic in ``spec.seed``: the same spec always renders the same
    samples, which is what makes the whole evaluation reproducible.
    """
    rng = np.random.default_rng(spec.seed)
    n = int(round(duration_s * sample_rate))
    out = np.zeros(n, dtype=np.float64)
    nyquist = sample_rate / 2.0

    # --- note sequence: a random walk over the scale ---
    note_len = 1.0 / spec.notes_per_second
    n_notes = int(np.ceil(duration_s / note_len)) + 1
    degree = 0
    octave = 0
    for k in range(n_notes):
        start = int(round(k * note_len * sample_rate))
        if start >= n:
            break
        length = min(int(round(note_len * 1.6 * sample_rate)), n - start)
        if length <= 8:
            continue
        degree = int(np.clip(degree + rng.integers(-2, 3), 0, len(spec.scale) - 1))
        if rng.random() < 0.15:
            octave = int(np.clip(octave + rng.choice([-1, 1]), 0, 2))
        f0 = spec.root_hz * 2.0 ** ((spec.scale[degree] + 12 * octave) / 12.0)

        t = np.arange(length, dtype=np.float64) / sample_rate
        note = np.zeros(length, dtype=np.float64)
        for h, gain in enumerate(spec.harmonic_gains, start=1):
            fh = f0 * h
            if fh >= nyquist * 0.95:
                break
            phase = float(rng.uniform(0, 2 * np.pi))
            note += gain * np.sin(2 * np.pi * fh * t + phase)
        note *= _adsr(length, sample_rate)
        out[start : start + length] += note

        if spec.click_level > 0:
            click_n = min(int(0.004 * sample_rate), n - start)
            if click_n > 0:
                click = rng.standard_normal(click_n) * spec.click_level
                click *= np.linspace(1.0, 0.0, click_n) ** 2
                out[start : start + click_n] += click

    # --- sustained pad partials ---
    t_full = np.arange(n, dtype=np.float64) / sample_rate
    for f in spec.pad_partials_hz:
        if f < nyquist * 0.95:
            out += 0.08 * np.sin(2 * np.pi * f * t_full + rng.uniform(0, 2 * np.pi))

    rms = float(np.sqrt(np.mean(out**2)))
    if rms > 0:
        out *= 0.1 / rms
    return out


@dataclass
class Corpus:
    """A rendered SYNTHETIC corpus.

    Attributes:
        specs: Track specifications.
        audio: ``track_id -> signal``.
        sample_rate: Sample rate of every signal.
    """

    specs: list[TrackSpec]
    audio: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    sample_rate: int = DEFAULT_CONFIG.sample_rate

    def __len__(self) -> int:
        return len(self.specs)


def make_corpus(
    n_tracks: int,
    duration_s: float = 30.0,
    sample_rate: int = DEFAULT_CONFIG.sample_rate,
    seed: int = 0,
    *,
    id_offset: int = 0,
) -> Corpus:
    """Generate and render ``n_tracks`` SYNTHETIC tracks.

    Args:
        n_tracks: How many to generate.
        duration_s: Length of each track.
        sample_rate: Output rate.
        seed: Master seed; the whole corpus is a deterministic function of it.
        id_offset: Added to every ``track_id``. Used to mint an impostor set
            whose ids cannot collide with the indexed corpus — a collision there
            would silently turn a false accept into a "correct" answer and
            flatter the results.
    """
    rng = np.random.default_rng(seed)
    specs = [make_track_spec(i + id_offset, rng) for i in range(n_tracks)]
    corpus = Corpus(specs=specs, sample_rate=sample_rate)
    for spec in specs:
        corpus.audio[spec.track_id] = synthesize(spec, duration_s, sample_rate)
    return corpus


# --------------------------------------------------------------------------
# Degradations
# --------------------------------------------------------------------------


def measured_snr_db(clean: NDArray[np.float64], noisy: NDArray[np.float64]) -> float:
    """Measure the SNR of ``noisy`` treating ``noisy - clean`` as the noise.

    Exists so the evaluation can *verify* that ``add_noise`` produced the SNR it
    was asked for, rather than assuming it. An SNR sweep whose x-axis is wrong is
    worse than no sweep at all.
    """
    noise = np.asarray(noisy, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    p_sig = float(np.mean(np.asarray(clean, dtype=np.float64) ** 2))
    p_noise = float(np.mean(noise**2))
    if p_noise <= 0:
        return float("inf")
    return 10.0 * np.log10(p_sig / p_noise)


def add_noise(
    x: NDArray[np.float64], snr_db: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Add white Gaussian noise at a specified signal-to-noise ratio.

    The noise level is derived from the *measured* power of ``x``, not from an
    assumed nominal level, so the requested SNR is the achieved SNR regardless of
    how loud the excerpt is. White noise is the honest worst case for a peak
    picker: it raises the floor uniformly, so it attacks high-frequency peaks
    (which are weakest in real signals) hardest. Real interference — a café, a
    car, a crowd — is spectrally shaped and usually concentrated at low
    frequency, which a banded peak picker handles *better* than white noise.
    """
    x = np.asarray(x, dtype=np.float64)
    p_sig = float(np.mean(x**2))
    if p_sig <= 0:
        return x.copy()
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return x + rng.standard_normal(x.size) * np.sqrt(p_noise)


def band_limit(
    x: NDArray[np.float64],
    sample_rate: int,
    low_hz: float | None = None,
    high_hz: float | None = None,
    order: int = 4,
) -> NDArray[np.float64]:
    """Butterworth band-pass / high-pass / low-pass.

    Uses ``sosfiltfilt`` (second-order sections, zero-phase) rather than
    ``lfilter``. Zero-phase matters here: a causal filter's group delay would
    shift the query in time relative to the reference, and while the offset vote
    would absorb a *constant* shift, a frequency-dependent one smears peaks
    across frames and would make the band-limit results look worse than the
    band-limiting itself warrants. SOS rather than transfer-function form
    because a high-order Butterworth in ``(b, a)`` form is numerically unstable.

    Models: a phone microphone's response, telephone-band transmission, or the
    limited bandwidth of a worn stylus.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = sample_rate / 2.0
    if low_hz is None and high_hz is None:
        return x.copy()
    if low_hz is not None and high_hz is not None:
        sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    elif low_hz is not None:
        sos = butter(order, low_hz / nyq, btype="highpass", output="sos")
    else:
        assert high_hz is not None
        sos = butter(order, high_hz / nyq, btype="lowpass", output="sos")
    return np.asarray(sosfiltfilt(sos, x), dtype=np.float64)


def crop(
    x: NDArray[np.float64],
    sample_rate: int,
    seconds: float,
    rng: np.random.Generator,
    *,
    margin_s: float = 0.5,
) -> tuple[NDArray[np.float64], float]:
    """Cut a random excerpt of ``seconds`` from ``x``.

    Returns:
        ``(excerpt, start_seconds)``. The start time is returned so the
        evaluation can check the *alignment* the matcher recovers, not just the
        track id. Getting the track right by luck and the offset wrong is a
        different failure from getting both right, and worth being able to tell
        apart.

    ``margin_s`` keeps the excerpt away from the very start and end, where the
    synthesis envelope is atypical.
    """
    x = np.asarray(x, dtype=np.float64)
    n_want = int(round(seconds * sample_rate))
    if n_want >= x.size:
        return x.copy(), 0.0
    margin = int(margin_s * sample_rate)
    hi = max(margin + 1, x.size - n_want - margin)
    start = int(rng.integers(margin, hi))
    return x[start : start + n_want].copy(), start / sample_rate


def apply_gain(x: NDArray[np.float64], gain_db: float) -> NDArray[np.float64]:
    """Scale by ``gain_db`` decibels.

    Should be a no-op for retrieval: the dB spectrogram plus a local-statistics
    threshold makes peak picking exactly gain-invariant. The evaluation includes
    it precisely to check that claim empirically rather than trusting the
    argument.
    """
    return np.asarray(x, dtype=np.float64) * (10.0 ** (gain_db / 20.0))


def speed_change(x: NDArray[np.float64], rate: float, max_denom: int = 2000) -> NDArray[np.float64]:
    """Resample by ``rate`` — pitch and tempo change together.

    ``rate > 1`` plays faster and higher. This models a **turntable speed
    error**: a platter running 0.5% fast raises every partial by 0.5% and
    shortens every interval by the same factor. That is the realistic failure
    mode for a needle-drop rip, and it is nastier for fingerprinting than pure
    tempo change because it moves peaks in *both* axes at once.
    """
    frac = Fraction(rate).limit_denominator(max_denom)
    return np.asarray(
        resample_poly(np.asarray(x, dtype=np.float64), frac.denominator, frac.numerator),
        dtype=np.float64,
    )


def time_stretch(
    x: NDArray[np.float64],
    rate: float,
    *,
    frame_length: int = 1024,
    synthesis_hop: int = 256,
) -> NDArray[np.float64]:
    """Overlap-add time stretch: tempo changes, pitch does not.

    ``rate > 1`` shortens the signal. Implemented as plain OLA with a Hann
    window and 75% synthesis overlap, with no phase correction — so it does
    introduce some phasiness. That is stated rather than hidden: a phase vocoder
    would be cleaner, and the roadmap lists it. What OLA does preserve is the
    *frequency* of every partial, which is the quantity the fingerprint depends
    on, so it is an adequate model of a broadcast time-compressor.
    """
    x = np.asarray(x, dtype=np.float64)
    analysis_hop = max(1, int(round(synthesis_hop * rate)))
    win = hann_window(frame_length)
    n_frames = max(1, 1 + (x.size - frame_length) // analysis_hop)
    out_len = (n_frames - 1) * synthesis_hop + frame_length
    out = np.zeros(out_len, dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)
    for i in range(n_frames):
        a = i * analysis_hop
        s = i * synthesis_hop
        out[s : s + frame_length] += x[a : a + frame_length] * win
        norm[s : s + frame_length] += win
    return out / np.maximum(norm, 1e-8)


def make_impostor_corpus(
    n_tracks: int,
    duration_s: float = 30.0,
    sample_rate: int = DEFAULT_CONFIG.sample_rate,
    seed: int = 12345,
    *,
    id_offset: int = 100_000,
) -> Corpus:
    """SYNTHETIC tracks generated the same way but never added to the index.

    Queries drawn from these are the only way to measure a false-accept rate:
    without them you can measure "how often is the top-1 correct" but not "how
    often does the system confidently name a track for audio it has never seen",
    which is the failure mode that actually matters in deployment. A system that
    is 99% accurate on in-corpus queries and names something with high confidence
    for every out-of-corpus query is not usable.
    """
    return make_corpus(
        n_tracks, duration_s=duration_s, sample_rate=sample_rate, seed=seed,
        id_offset=id_offset,
    )


def spectrogram_config_for(sample_rate: int) -> SpectrogramConfig:
    """Analysis geometry matched to a sample rate (keeps hop/FFT defaults)."""
    return SpectrogramConfig(
        sample_rate=sample_rate,
        n_fft=DEFAULT_CONFIG.n_fft,
        hop_length=DEFAULT_CONFIG.hop_length,
    )
