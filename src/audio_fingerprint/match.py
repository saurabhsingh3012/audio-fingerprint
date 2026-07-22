"""Retrieval by time-offset histogram voting.

Why offset consistency is the discriminative signal — not hash overlap
---------------------------------------------------------------------
The tempting score is "how many hashes does the query share with track X". It is
also the wrong one, and understanding why is the difference between a demo and a
retrieval system.

Two unrelated pieces of music share a lot of local spectral geometry. Both
contain notes; notes have harmonics; harmonics produce peak pairs with similar
frequency relationships and similar time gaps. Over a query generating a few
thousand hashes, chance collisions accumulate steadily — and they accumulate
*fastest* for whichever reference track is longest or densest, because it simply
has more postings. Raw hash overlap therefore carries a systematic bias towards
long, busy tracks and no bias at all towards the correct one.

What the correct track has that impostors do not is **structure in the
collisions**. If the query is the segment starting 47.2 s into track X, then
*every* genuine matching hash satisfies

    reference_anchor_time - query_anchor_time = 47.2 s

with the same constant. Chance collisions carry no such constraint: their
differences spread roughly uniformly over the reference's duration. So the
histogram of ``Δ = ref_offset - query_offset`` shows a sharp spike on one bin
for the correct track, and a low flat smear for an impostor.

Scoring on **the height of the tallest bin** rather than the total does three
things at once:

1. It is a matched filter for the thing that actually distinguishes a true
   match, so genuine and impostor score distributions separate far more than
   raw counts do. This project measures the gap rather than asserting it (see
   the README table, and ``tests/test_matching.py`` for a constructed
   near-miss where raw counting demonstrably picks the wrong track).
2. It removes the long-track bias: a longer reference spreads its chance
   collisions over more bins, so its tallest *random* bin barely grows even as
   its total collision count does.
3. It recovers the alignment for free. The winning bin *is* the position of the
   excerpt within the track — which is what makes a needle-drop rip locatable
   against a reference pressing, not merely identifiable.

Tolerance, and why the histogram is smoothed
--------------------------------------------
Δ is quantised to frames. A query resampled by 0.5% — a turntable running
slightly fast, or a broadcast pitch-shift — accumulates drift across the
excerpt, so its matching hashes land in a small *cluster* of neighbouring bins
rather than exactly one. Summing over ``±tolerance_bins`` recovers those votes.
The window is kept narrow (default ±1 frame = ±23 ms at the default geometry)
because widening it also sums more of the impostor smear, which is exactly the
noise being rejected. The smoothing is applied identically to genuine and
impostor queries, so the reported separation is not an artefact of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hashing import Fingerprint
from .index import FingerprintIndex
from .spectrogram import DEFAULT_CONFIG, SpectrogramConfig

__all__ = [
    "MatchConfig",
    "MatchResult",
    "identify",
    "rank_by_raw_hash_count",
]


@dataclass(frozen=True)
class MatchConfig:
    """Voting parameters.

    Args:
        tolerance_bins: Half-width, in frames, of the histogram smoothing
            window.
        max_posting_length: Query-time cut on posting-list length; ``None``
            disables it. Off by default because no key in this project's
            synthetic corpus is pathological, but it is the first knob to reach
            for when queries get slow on a real corpus.
        top_k: Number of ranked candidates to return.
    """

    tolerance_bins: int = 1
    max_posting_length: int | None = None
    top_k: int = 5


DEFAULT_MATCH_CONFIG = MatchConfig()


@dataclass(frozen=True)
class MatchResult:
    """One candidate track and its evidence.

    Attributes:
        track_id: Candidate identifier.
        name: Candidate display name.
        score: Height of the tallest smoothed offset-histogram bin. **This is
            the retrieval score.**
        offset_frames: Δ of the winning bin — the query's estimated position
            inside the reference, in frames.
        offset_seconds: The same, in seconds.
        raw_hits: Total hash collisions with this track, ignoring alignment.
            Reported for diagnostics and for the head-to-head against voting.
        n_query_hashes: Size of the query fingerprint, so ``score`` can be
            normalised across excerpt lengths.
    """

    track_id: int
    name: str
    score: float
    offset_frames: int
    offset_seconds: float
    raw_hits: int
    n_query_hashes: int

    @property
    def normalised_score(self) -> float:
        """``score / n_query_hashes``.

        A 10-second query produces roughly ten times the hashes of a 1-second
        one, so its absolute score is ~10x larger for the same *quality* of
        match. Any fixed absolute accept threshold therefore implicitly demands
        a much better match from short queries than from long ones. The
        normalised score removes that coupling — at the cost of discarding the
        genuine information that a long query is intrinsically more trustworthy.
        This project reports false-accept rates under both.
        """
        return self.score / max(self.n_query_hashes, 1)


def _histogram_votes(
    track_ids: NDArray[np.int32],
    deltas: NDArray[np.int32],
    tolerance_bins: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Vectorised per-track offset histogram.

    Returns ``(unique_tracks, best_score, best_delta, raw_hits)``.

    Implemented as a single ``bincount`` over a flattened ``(track, delta)``
    index rather than a loop over tracks, because the evaluation grid issues
    ~1,000 queries and a Python loop per candidate track would dominate runtime.
    """
    uniq, track_pos = np.unique(track_ids, return_inverse=True)
    n_tracks = uniq.size
    lo = int(deltas.min())
    hi = int(deltas.max())
    n_bins = hi - lo + 1

    flat = track_pos.astype(np.int64) * n_bins + (deltas.astype(np.int64) - lo)
    hist = np.bincount(flat, minlength=n_tracks * n_bins).reshape(n_tracks, n_bins)

    width = 2 * tolerance_bins + 1
    if width > 1:
        # Sliding-window sum along the delta axis via a prefix sum: O(n) and
        # fully vectorised, versus a per-track convolve loop.
        padded = np.pad(hist, ((0, 0), (tolerance_bins, tolerance_bins)))
        csum = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)))
        smoothed = csum[:, width:] - csum[:, :-width]
    else:
        smoothed = hist

    best_bin = smoothed.argmax(axis=1)
    best_score = smoothed[np.arange(n_tracks), best_bin]
    best_delta = best_bin.astype(np.int64) + lo
    raw_hits = hist.sum(axis=1)
    return uniq.astype(np.int64), best_score.astype(np.int64), best_delta, raw_hits


def identify(
    index: FingerprintIndex,
    fp: Fingerprint,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
    spec_config: SpectrogramConfig = DEFAULT_CONFIG,
) -> list[MatchResult]:
    """Rank candidate tracks for a query fingerprint.

    Args:
        index: Reference index.
        fp: Query fingerprint.
        config: Voting parameters.
        spec_config: Analysis geometry, used only to convert the winning frame
            offset to seconds.

    Returns:
        Up to ``config.top_k`` results, sorted by descending score. Ties break on
        ``raw_hits`` then ``track_id`` so ranking is deterministic — an unstable
        tie-break makes accuracy numbers irreproducible run to run, which is a
        silent way to make an evaluation meaningless.

        An empty list means the query produced no hash collisions at all. That
        is itself a useful signal: an out-of-corpus recording, or an excerpt too
        short or too degraded to fingerprint.
    """
    if len(fp) == 0:
        return []
    track_ids, deltas = index.lookup_many(
        fp.hashes, fp.offsets, max_posting_length=config.max_posting_length
    )
    if track_ids.size == 0:
        return []

    uniq, scores, best_deltas, raw_hits = _histogram_votes(
        track_ids, deltas, config.tolerance_bins
    )
    results = [
        MatchResult(
            track_id=int(t),
            name=index.track_names.get(int(t), str(int(t))),
            score=float(s),
            offset_frames=int(d),
            offset_seconds=spec_config.frames_to_seconds(int(d)),
            raw_hits=int(r),
            n_query_hashes=len(fp),
        )
        for t, s, d, r in zip(uniq, scores, best_deltas, raw_hits, strict=True)
    ]
    results.sort(key=lambda r: (-r.score, -r.raw_hits, r.track_id))
    return results[: config.top_k]


def rank_by_raw_hash_count(
    index: FingerprintIndex,
    fp: Fingerprint,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
) -> list[MatchResult]:
    """Baseline ranker that ignores alignment and counts collisions only.

    Kept in the library rather than buried in a test so the comparison is
    reproducible and the README's claim about offset voting is falsifiable by
    anyone who clones the repo. This is the ranker the system deliberately does
    *not* use.
    """
    if len(fp) == 0:
        return []
    track_ids, _ = index.lookup_many(
        fp.hashes, fp.offsets, max_posting_length=config.max_posting_length
    )
    if track_ids.size == 0:
        return []
    uniq, counts = np.unique(track_ids, return_counts=True)
    results = [
        MatchResult(
            track_id=int(t),
            name=index.track_names.get(int(t), str(int(t))),
            score=float(c),
            offset_frames=0,
            offset_seconds=0.0,
            raw_hits=int(c),
            n_query_hashes=len(fp),
        )
        for t, c in zip(uniq, counts, strict=True)
    ]
    results.sort(key=lambda r: (-r.score, r.track_id))
    return results[: config.top_k]
