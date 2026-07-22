"""Short-time Fourier analysis, written from scratch on numpy/scipy primitives.

Everything here is deliberately hand-rolled (framing, windowing, mel filterbank)
rather than pulled from ``librosa``. The point of the project is to show the
signal-processing reasoning, and a wrapper around someone else's ``melspectrogram``
call hides exactly the decisions that matter for fingerprint robustness.

Why a *linear-frequency* STFT is the default for fingerprinting
--------------------------------------------------------------
Mel scaling is the right default for *perceptual* tasks (speech recognition,
music tagging) because it spends resolution where hearing is sensitive and
throws it away above ~4 kHz. Fingerprinting is not a perceptual task. It needs
*sharp, repeatable* spectral landmarks, and a mel filterbank deliberately blurs
neighbouring bins together at high frequency — which is precisely where a
fingerprint wants its most discriminative, least-crowded peaks to live. Blurring
also makes a peak's bin index depend on how much energy landed in the
neighbouring bins, which is exactly what additive noise perturbs.

So: mel scaling is provided (:func:`mel_filterbank`, :func:`melspectrogram`) and
is useful for visualisation and for the noise-floor discussion, but the
fingerprint pipeline runs on the linear STFT.

Why decibels rather than linear power
-------------------------------------
Every downstream stage (peak picking, thresholding) becomes gain-invariant once
the magnitude is in dB, because a gain change of ``g`` maps to an *additive*
constant ``20*log10(g)`` across the whole spectrogram. Local maxima are unmoved
by adding a constant, and a "local mean + k * local std" threshold is unmoved as
well. That single choice is what makes the fingerprint survive a volume knob,
and it is why :func:`amplitude_to_db` uses an *absolute* floor rather than
normalising by the per-track maximum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft
from numpy.typing import NDArray

__all__ = [
    "SpectrogramConfig",
    "amplitude_to_db",
    "fft_frequencies",
    "frame_signal",
    "frame_times",
    "hann_window",
    "hz_to_mel",
    "log_power_spectrogram",
    "mel_filterbank",
    "mel_to_hz",
    "melspectrogram",
    "power_spectrogram",
    "stft",
]

#: Absolute power floor used before taking a logarithm. Chosen at -200 dB so
#: that it never truncates real signal content at the gain levels we test
#: (down to -40 dB), which would silently break gain invariance.
EPS_POWER: float = 1e-20


@dataclass(frozen=True)
class SpectrogramConfig:
    """Analysis geometry shared by every stage of the pipeline.

    Keeping this in one frozen object matters more than it looks: a fingerprint
    hash encodes *frequency bin indices* and *frame deltas*, so a reference
    database and a query are only comparable if they were analysed with
    identical ``n_fft`` and ``hop_length``. Passing the config around by value
    makes an accidental mismatch a type error rather than a silent accuracy
    collapse.

    Args:
        sample_rate: Working sample rate in Hz. The pipeline downsamples to a
            low rate on purpose (see ``DEFAULT_CONFIG``).
        n_fft: FFT length in samples; also the analysis frame length.
        hop_length: Advance between successive frames in samples.
    """

    sample_rate: int = 11025
    n_fft: int = 1024
    hop_length: int = 256

    @property
    def n_bins(self) -> int:
        """Number of non-negative-frequency bins produced by :func:`stft`."""
        return self.n_fft // 2 + 1

    @property
    def frames_per_second(self) -> float:
        """Frame rate of the spectrogram, in frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def bin_hz(self) -> float:
        """Width of one frequency bin, in Hz."""
        return self.sample_rate / self.n_fft

    def frames_to_seconds(self, frames: float) -> float:
        """Convert a frame count (or frame delta) to seconds."""
        return float(frames) * self.hop_length / self.sample_rate


#: Default analysis geometry.
#:
#: ``sample_rate=11025`` because fingerprinting does not need full bandwidth:
#: the landmarks that survive a noisy microphone recording live below ~5 kHz
#: anyway, and quartering the sample rate quarters the FFT cost. Real systems
#: use a comparable rate for the same reason.
#:
#: ``n_fft=1024`` gives 10.8 Hz bins and a 93 ms window — long enough to resolve
#: individual partials of a bass note, short enough that a note onset is not
#: smeared across the whole window. ``hop_length=256`` (75% overlap) gives 43
#: frames/s, so a 1-second query still yields ~43 frames of evidence.
DEFAULT_CONFIG = SpectrogramConfig()


def hann_window(length: int, *, periodic: bool = True) -> NDArray[np.float64]:
    """Return a Hann window.

    Args:
        length: Window length in samples.
        periodic: If True (default) use the DFT-periodic form ``1/N``, which is
            the correct choice for spectral analysis — the symmetric ``1/(N-1)``
            form is for filter design and introduces a small bias in the
            overlap-add sum.

    Returns:
        Array of shape ``(length,)``.

    Why Hann and not a flat rectangular window: a rectangular window's spectral
    leakage sidelobes are only -13 dB down, which sprays energy from a loud
    partial across dozens of bins and manufactures spurious "peaks" beside every
    real one. Hann's -31 dB sidelobes keep the constellation sparse and honest,
    at the cost of a slightly wider main lobe.
    """
    if length <= 0:
        raise ValueError(f"window length must be positive, got {length}")
    if length == 1:
        return np.ones(1, dtype=np.float64)
    denom = length if periodic else length - 1
    k = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / denom)


def frame_signal(
    x: NDArray[np.float64], frame_length: int, hop_length: int
) -> NDArray[np.float64]:
    """Slice a 1-D signal into overlapping frames.

    Args:
        x: Mono signal, shape ``(n_samples,)``.
        frame_length: Samples per frame.
        hop_length: Advance between frames.

    Returns:
        Array of shape ``(n_frames, frame_length)``. Trailing samples that do
        not fill a whole frame are dropped.

    Raises:
        ValueError: If the signal is shorter than one frame. This is raised
            rather than zero-padded because a caller asking to fingerprint less
            than 93 ms of audio has a bug, and silently returning one mostly-zero
            frame would produce a plausible-looking but meaningless fingerprint.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a mono 1-D signal, got shape {x.shape}")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if x.size < frame_length:
        raise ValueError(
            f"signal of {x.size} samples is shorter than one frame "
            f"({frame_length} samples)"
        )
    n_frames = 1 + (x.size - frame_length) // hop_length
    strided = np.lib.stride_tricks.sliding_window_view(x, frame_length)
    # sliding_window_view returns a read-only view; copy so downstream code can
    # multiply by the window in place without a hidden aliasing bug.
    return np.array(strided[:: hop_length][:n_frames], dtype=np.float64, copy=True)


def stft(
    x: NDArray[np.float64],
    config: SpectrogramConfig = DEFAULT_CONFIG,
    *,
    window: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """Short-time Fourier transform, ``(n_bins, n_frames)`` complex.

    Args:
        x: Mono signal.
        config: Analysis geometry.
        window: Optional analysis window of length ``config.n_fft``. Defaults to
            a periodic Hann window.

    Returns:
        Complex array of shape ``(config.n_bins, n_frames)``.

    Note:
        There is no centring / reflect-padding. Padding shifts every frame index
        by ``n_fft / (2 * hop)`` frames, and because a fingerprint hash encodes
        *time deltas between peaks*, a constant shift is harmless — but only if
        reference and query agree. Not padding at all removes the chance of
        disagreeing.
    """
    frames = frame_signal(x, config.n_fft, config.hop_length)
    win = hann_window(config.n_fft) if window is None else np.asarray(window)
    if win.shape != (config.n_fft,):
        raise ValueError(f"window must have shape ({config.n_fft},), got {win.shape}")
    # scipy.fft rather than numpy.fft: measured ~4x faster here with multiple
    # workers, and the result is bit-identical because each frame's transform is
    # independent (workers only parallelise across rows, never within one).
    spectra = scipy.fft.rfft(frames * win, n=config.n_fft, axis=1, workers=-1)
    return np.ascontiguousarray(spectra.T)


def power_spectrogram(
    x: NDArray[np.float64],
    config: SpectrogramConfig = DEFAULT_CONFIG,
) -> NDArray[np.float64]:
    """Magnitude-squared STFT, shape ``(n_bins, n_frames)``."""
    spec = stft(x, config)
    return np.real(spec * np.conjugate(spec))


def amplitude_to_db(
    power: NDArray[np.float64], *, amin: float = EPS_POWER
) -> NDArray[np.float64]:
    """Convert a power spectrogram to decibels against an absolute reference.

    Args:
        power: Non-negative power values.
        amin: Absolute floor applied before the log.

    Returns:
        ``10 * log10(max(power, amin))``.

    Why no per-track max normalisation: normalising by the maximum couples every
    bin to whatever the loudest moment of the *excerpt* happened to be. A query
    that clips a loud chorus and a query that clips a quiet intro would then get
    different dB values for identical content, and the adaptive peak threshold
    would fire differently. An absolute reference keeps the two consistent, and
    gain invariance is recovered later by thresholding *relative to a local
    neighbourhood* instead.
    """
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), amin))


def log_power_spectrogram(
    x: NDArray[np.float64],
    config: SpectrogramConfig = DEFAULT_CONFIG,
) -> NDArray[np.float64]:
    """Convenience: signal to dB-scaled linear-frequency spectrogram."""
    return amplitude_to_db(power_spectrogram(x, config))


def fft_frequencies(config: SpectrogramConfig = DEFAULT_CONFIG) -> NDArray[np.float64]:
    """Centre frequency in Hz of each STFT bin."""
    return np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)


def frame_times(
    n_frames: int, config: SpectrogramConfig = DEFAULT_CONFIG
) -> NDArray[np.float64]:
    """Start time in seconds of each analysis frame."""
    return np.arange(n_frames, dtype=np.float64) * config.hop_length / config.sample_rate


def hz_to_mel(hz: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """HTK mel scale: ``2595 * log10(1 + f/700)``."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    config: SpectrogramConfig = DEFAULT_CONFIG,
    *,
    n_mels: int = 64,
    fmin: float = 40.0,
    fmax: float | None = None,
) -> NDArray[np.float64]:
    """Triangular mel filterbank, shape ``(n_mels, config.n_bins)``.

    Filters are area-normalised (Slaney style) so that a flat-power input gives
    a flat mel spectrum instead of one that rises with frequency purely because
    high-frequency triangles are wider.

    Provided for visualisation and for the noise-floor analysis. It is **not**
    used by the fingerprint path — see the module docstring for why.
    """
    if fmax is None:
        fmax = config.sample_rate / 2.0
    if not 0 <= fmin < fmax <= config.sample_rate / 2.0:
        raise ValueError(f"require 0 <= fmin < fmax <= nyquist, got {fmin}..{fmax}")
    mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    bin_hz = fft_frequencies(config)

    bank = np.zeros((n_mels, config.n_bins), dtype=np.float64)
    for m in range(n_mels):
        left, centre, right = hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]
        rising = (bin_hz - left) / max(centre - left, 1e-12)
        falling = (right - bin_hz) / max(right - centre, 1e-12)
        tri = np.maximum(0.0, np.minimum(rising, falling))
        area = 0.5 * (right - left)
        bank[m] = tri * (2.0 / area) if area > 0 else tri
    return bank


def melspectrogram(
    x: NDArray[np.float64],
    config: SpectrogramConfig = DEFAULT_CONFIG,
    *,
    n_mels: int = 64,
    fmin: float = 40.0,
    fmax: float | None = None,
) -> NDArray[np.float64]:
    """dB-scaled mel spectrogram, shape ``(n_mels, n_frames)``."""
    power = power_spectrogram(x, config)
    bank = mel_filterbank(config, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return amplitude_to_db(bank @ power)
