"""Spectral peak picking: turning a spectrogram into a sparse constellation.

The whole fingerprinting idea rests on one observation: **spectral peaks are the
most noise-robust thing in a spectrogram.** Additive noise raises the floor
everywhere but rarely creates a new local maximum that beats a real partial, and
it never *moves* a strong partial. Band-limiting, MP3 coding and a cheap
microphone all destroy absolute levels while leaving the location of surviving
peaks intact. So we throw away amplitude and keep only *where* the peaks are.

The density / robustness trade-off
----------------------------------
Peak density (peaks per second retained) is the single most important tuning
knob in the system, and it is a genuine trade-off, not a "higher is better" dial:

* **Too sparse** (say 5 peaks/s). A short, noisy query may only reproduce half
  of the reference's peaks. With few peaks there are few hash pairs, so a 3-second
  query might generate only a handful of hashes and a single unlucky miss
  destroys the match. Recall collapses.
* **Too dense** (say 100 peaks/s). Recall improves, but every extra peak is a
  weaker peak — a noise-floor bump rather than a partial — so the *added* hashes
  are mostly noise. Database size and query cost grow roughly linearly in
  density, and the number of hash *pairs* grows with density x fan-out. Worse,
  dense constellations from two unrelated tracks start sharing hashes by chance,
  which raises the false-accept floor. Precision degrades.

The right operating point depends on corpus size and expected noise. The default
here — **30 peaks/s** — was chosen by measurement on a tuning corpus disjoint
from the evaluation corpus, and the choice is only visible at *hard* operating
points. At 3 s / 0 dB SNR, accuracy was flat (0.967-0.983) for every density
from 15 to 50 peaks/s, so an easy benchmark would have said "use 15 and save
two-thirds of the index". Pushing to 1 s excerpts at -5 dB SNR separated them:

    density   index postings   1 s/-5 dB   1.5 s/0 dB   5 s/-10 dB
       10           22,246       0.417       0.633        0.750
       20           62,993       0.583       0.817        0.833
       30          101,819       0.583       0.867        0.800
       45          159,822       0.633       0.917        0.800

10 peaks/s is clearly too sparse. Above 30 the returns are condition-dependent —
45 helps short excerpts and does not help low SNR — while the index grows
linearly. 30 is the knee. **The lesson worth keeping is that the density knob is
invisible until the evaluation includes conditions hard enough to break the
system**; tuning on easy queries would have picked a value that fails in the
field.

Making density actually *hit* the target
----------------------------------------
A naive "keep everything above a threshold" gives wildly varying density: loud
passages emit hundreds of peaks, quiet ones emit none. That is bad because
matching score is roughly proportional to the number of query hashes, so score
would depend on how loud the excerpt happened to be. Instead we enforce density
structurally:

1. local-maximum test (a peak must dominate its time-frequency neighbourhood),
2. adaptive threshold (must stand above the *local* noise floor),
3. quota selection over a grid of ``(1 second) x (frequency band)`` cells.

Step 3 is what pins the density and also what spreads peaks across the spectrum.
Without banding, a bass-heavy track puts every peak below 300 Hz, and a
high-pass-filtered query then matches nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter, uniform_filter

from .spectrogram import DEFAULT_CONFIG, SpectrogramConfig, log_power_spectrogram

__all__ = ["Constellation", "PeakConfig", "peak_density", "pick_peaks"]


@dataclass(frozen=True)
class PeakConfig:
    """Peak-picking parameters.

    Args:
        target_density: Peaks per second to retain. See module docstring.
        n_bands: Number of geometrically spaced frequency bands used for the
            quota. Geometric rather than linear because musical energy is
            roughly log-distributed in frequency; linear bands would give the
            bottom band all the content and the top five almost none.
        neighbourhood_freq: Height in bins of the local-maximum neighbourhood.
        neighbourhood_time: Width in frames of the local-maximum neighbourhood.
        background_freq: Height in bins of the window used to estimate the local
            noise floor.
        background_time: Width in frames of the same window.
        threshold_k: A candidate must exceed ``local_mean + threshold_k *
            local_std``. Higher is stricter.
        min_band_bin: Lowest STFT bin considered. Rumble, turntable motor noise
            and DC offset live below this and produce enormous, useless peaks.
    """

    target_density: float = 30.0
    n_bands: int = 6
    neighbourhood_freq: int = 11
    neighbourhood_time: int = 9
    background_freq: int = 41
    background_time: int = 31
    threshold_k: float = 1.0
    min_band_bin: int = 4


DEFAULT_PEAK_CONFIG = PeakConfig()


@dataclass(frozen=True)
class Constellation:
    """A sparse set of spectral peaks, sorted by time then frequency.

    Attributes:
        frames: Frame index of each peak, ``int32``, ascending.
        bins: STFT bin index of each peak, ``int32``.
        magnitudes_db: dB magnitude at each peak. Kept for diagnostics and
            plotting only — **the fingerprint never uses it**, because absolute
            level is the least reliable quantity in a degraded recording.
        n_frames: Total frames in the source spectrogram, needed to compute
            density.
        config: Analysis geometry the peaks were computed under.
    """

    frames: NDArray[np.int32]
    bins: NDArray[np.int32]
    magnitudes_db: NDArray[np.float64]
    n_frames: int
    config: SpectrogramConfig = DEFAULT_CONFIG

    def __len__(self) -> int:
        return int(self.frames.size)

    @property
    def duration_seconds(self) -> float:
        """Duration of the analysed audio, in seconds."""
        return self.n_frames * self.config.hop_length / self.config.sample_rate

    @property
    def density(self) -> float:
        """Retained peaks per second."""
        return len(self) / max(self.duration_seconds, 1e-9)


def _band_edges(n_bins: int, n_bands: int, min_bin: int) -> NDArray[np.int64]:
    """Geometrically spaced band edges over bin indices.

    Geometric spacing mirrors how musical energy is distributed: the interval
    50-100 Hz carries about as much perceptual and spectral structure as
    2000-4000 Hz, and linear bands would put ~90% of the bins in the top band.
    """
    lo = max(min_bin, 1)
    edges = np.geomspace(lo, n_bins, n_bands + 1)
    edges = np.unique(np.round(edges).astype(np.int64))
    # Degenerate geometry (tiny n_bins) can collapse edges; fall back to linear.
    if edges.size < n_bands + 1:
        edges = np.unique(np.linspace(lo, n_bins, n_bands + 1).astype(np.int64))
    return edges


def pick_peaks(
    audio_or_spec: NDArray[np.float64],
    config: SpectrogramConfig = DEFAULT_CONFIG,
    peak_config: PeakConfig = DEFAULT_PEAK_CONFIG,
    *,
    is_spectrogram: bool = False,
) -> Constellation:
    """Extract a sparse, density-controlled constellation of spectral peaks.

    Args:
        audio_or_spec: Mono audio, or a dB spectrogram if ``is_spectrogram``.
        config: Analysis geometry.
        peak_config: Peak-picking parameters.
        is_spectrogram: Treat the input as an already-computed dB spectrogram of
            shape ``(n_bins, n_frames)``.

    Returns:
        A :class:`Constellation`.

    The three stages, and why each is needed:

    **1. Local maximum.** ``S[f, t]`` must be the maximum over a
    ``neighbourhood_freq x neighbourhood_time`` box. This enforces minimum
    spacing so a single loud partial contributes *one* landmark rather than a
    smear of nine adjacent bins, which would make hashes redundant without
    making them more informative.

    **2. Adaptive threshold.** ``S[f,t] > local_mean + k * local_std`` over a
    much larger box. A *global* threshold fails immediately: it is a level test,
    so it survives neither a gain change nor a track with a quiet intro. The
    local statistics are computed in dB, so a gain change adds the same constant
    to ``S`` and to ``local_mean`` and leaves ``local_std`` untouched — the test
    is exactly gain-invariant. That is the whole reason for working in dB.

    **3. Per-cell quota.** Time is cut into 1-second blocks and frequency into
    ``n_bands`` geometric bands; within each cell we keep the strongest
    ``round(target_density / n_bands)`` candidates. If a block cannot fill its
    quota (silence, or an aggressively band-limited query) the remaining slots
    are refilled from the best unused candidates elsewhere in that same block, so
    density stays near target instead of collapsing.
    """
    if is_spectrogram:
        spec_db = np.asarray(audio_or_spec, dtype=np.float64)
        if spec_db.ndim != 2:
            raise ValueError(f"expected a 2-D spectrogram, got {spec_db.shape}")
    else:
        spec_db = log_power_spectrogram(audio_or_spec, config)

    n_bins, n_frames = spec_db.shape
    if n_frames == 0:
        empty_i = np.zeros(0, dtype=np.int32)
        return Constellation(empty_i, empty_i, np.zeros(0), 0, config)

    # --- Stage 1: local maxima -------------------------------------------------
    local_max = maximum_filter(
        spec_db,
        size=(peak_config.neighbourhood_freq, peak_config.neighbourhood_time),
        mode="nearest",
    )
    is_peak = spec_db >= local_max

    # --- Stage 2: adaptive threshold ------------------------------------------
    size = (peak_config.background_freq, peak_config.background_time)
    mean = uniform_filter(spec_db, size=size, mode="nearest")
    mean_sq = uniform_filter(spec_db * spec_db, size=size, mode="nearest")
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    is_peak &= spec_db > mean + peak_config.threshold_k * std

    # Discard the very bottom of the spectrum outright.
    is_peak[: peak_config.min_band_bin, :] = False

    cand_bins, cand_frames = np.nonzero(is_peak)
    if cand_bins.size == 0:
        empty_i = np.zeros(0, dtype=np.int32)
        return Constellation(empty_i, empty_i, np.zeros(0), n_frames, config)
    cand_db = spec_db[cand_bins, cand_frames]

    # --- Stage 3: quota selection ---------------------------------------------
    block_frames = max(1, int(round(config.frames_per_second)))
    n_blocks = int(n_frames // block_frames) + 1
    edges = _band_edges(n_bins, peak_config.n_bands, peak_config.min_band_bin)
    n_bands = edges.size - 1

    per_cell = max(1, int(round(peak_config.target_density / n_bands)))
    per_block = max(1, int(round(peak_config.target_density * block_frames
                                 / config.frames_per_second)))

    block_id = (cand_frames // block_frames).astype(np.int64)
    band_id = np.clip(np.searchsorted(edges, cand_bins, side="right") - 1, 0, n_bands - 1)

    # Sort once by (block, band, -magnitude) so quotas are a simple group scan.
    order = np.lexsort((-cand_db, band_id, block_id))
    keep = np.zeros(cand_bins.size, dtype=bool)

    cell_key = block_id[order] * n_bands + band_id[order]
    # Rank within each (block, band) cell.
    cell_change = np.empty(cell_key.size, dtype=bool)
    cell_change[0] = True
    cell_change[1:] = cell_key[1:] != cell_key[:-1]
    cell_start = np.maximum.accumulate(np.where(cell_change, np.arange(cell_key.size), 0))
    rank_in_cell = np.arange(cell_key.size) - cell_start
    keep[order] = rank_in_cell < per_cell

    # Refill: any block short of its overall quota takes its next-best leftovers.
    # `order` is lexsorted with block as the primary key, so block runs are
    # contiguous and their boundaries come from one searchsorted rather than a
    # full scan per block.
    block_of_sorted = block_id[order]
    bounds = np.searchsorted(block_of_sorted, np.arange(n_blocks + 1))
    for b in range(n_blocks):
        lo_b, hi_b = int(bounds[b]), int(bounds[b + 1])
        if hi_b <= lo_b:
            continue
        idx = order[lo_b:hi_b]
        n_kept = int(keep[idx].sum())
        deficit = per_block - n_kept
        if deficit <= 0:
            continue
        leftover = idx[~keep[idx]]
        if leftover.size == 0:
            continue
        best = leftover[np.argsort(-cand_db[leftover], kind="stable")[:deficit]]
        keep[best] = True

    sel_frames = cand_frames[keep].astype(np.int32)
    sel_bins = cand_bins[keep].astype(np.int32)
    sel_db = cand_db[keep]

    final_order = np.lexsort((sel_bins, sel_frames))
    return Constellation(
        frames=np.ascontiguousarray(sel_frames[final_order]),
        bins=np.ascontiguousarray(sel_bins[final_order]),
        magnitudes_db=np.ascontiguousarray(sel_db[final_order]),
        n_frames=int(n_frames),
        config=config,
    )


def peak_density(constellation: Constellation) -> float:
    """Peaks per second in ``constellation`` (convenience wrapper)."""
    return constellation.density
