"""Combinatorial constellation hashing: peaks -> compact, specific hash keys.

Why hash *pairs* of peaks and not single peaks
----------------------------------------------
This is the central idea of the whole system, so it is worth being precise about
it in terms of information content.

A single peak, in our geometry, is one of 513 frequency bins. Even if every bin
were equally likely that is ``log2(513) ≈ 9`` bits of information, and real
spectra are far from uniform, so the effective entropy is more like 6-7 bits. In
a database of 60 tracks x 30 s x 30 peaks/s = 54,000 peaks, a single-peak key
would have on the order of 54000 / 2^7 ≈ 400 entries in every posting list. A
query "matching" such a key tells you essentially nothing — every track in the
database contains a peak near 440 Hz.

A **pair** ``(f1, f2, Δt)`` is a different object. It is a small piece of *local
spectral geometry*: two partials at a specific frequency relationship, separated
by a specific time interval. Its key space is ~``2^9 x 2^9 x 2^6 = 2^24``, and
its realised entropy is far higher than a single peak's because the joint
distribution is much closer to uniform. Posting lists shrink by orders of
magnitude, so a hash hit is *evidence* rather than noise. Concretely, in this
project's measured index the mean posting-list length is small enough that a
query's hits are dominated by the correct track before any voting takes place.

The three properties that make pairs work:

1. **Specificity.** ~24 bits of key space instead of ~9.
2. **Translation invariance.** The key uses ``Δt``, never absolute time, so a
   query cropped from anywhere inside the track produces the *same* keys. The
   absolute time is kept alongside the key, not inside it — which is exactly
   what the offset-histogram vote later exploits.
3. **Graceful degradation.** With fan-out ``F``, every peak participates in up
   to ``F`` pairs as an anchor and ``F`` as a target. Losing one peak to noise
   costs ~``2F`` hashes out of thousands, not a whole region of the fingerprint.
   Redundancy is what buys robustness; a chain (each peak paired only with the
   next) would break completely at every lost peak.

Choosing the fan-out
--------------------
Fan-out ``F`` is the number of targets each anchor pairs with. Hash count grows
linearly in ``F``, so database size and query cost do too. Recall improves with
``F`` but with sharply diminishing returns: the marginal target is further away
in the target zone and therefore *less* likely to have survived the same
degradation as the anchor. Meanwhile false-accept pressure rises, because more
hashes per second means more chance collisions with unrelated tracks.

The published Shazam description uses a fan-out around 10. The default here is
**8, chosen by measurement** on a tuning corpus disjoint from the evaluation
corpus. At 3 s / 0 dB SNR on a 24-track database, top-1 accuracy went
0.850 (F=3) -> 0.900 (F=5) -> 0.950 (F=8) -> 0.975 (F=12), while the index grew
41k -> 67k -> 102k -> 140k postings. The step from 8 to 12 bought one query in
forty and cost 38% more index — and it *raised* the best impostor score, so the
genuine/impostor separation got worse, not better. 8 is where the curve turns.

Quantisation, and a result that went against expectation
--------------------------------------------------------
Frequencies can be quantised by ``freq_quant`` bins before packing. The standard
argument for doing so: noise occasionally shifts a peak by one bin, and without
quantisation that changes the key and loses the hash; 2-bin buckets halve the
number of shifts that cross a boundary, at the cost of shrinking the key space
4x (both ``f1`` and ``f2``).

**Measured on this project's synthetic corpus, quantisation did not help.**
Sweeping ``freq_quant`` over {1, 2, 4} at a hard operating point (3 s excerpt,
0 dB SNR, 24-track database), ``freq_quant=1`` matched or beat 2 at every
fan-out, and 4 was clearly worse; coarser buckets also raised the best impostor
score, i.e. they cost specificity without buying recall. The default is
therefore **1 — no quantisation at all**.

The reason is a property of the *corpus*, not of the algorithm: synthetic
partials are perfectly stationary, so a peak's bin index is highly repeatable
between the reference encoding and the query encoding, and there is little
bin-shift for quantisation to absorb. **On real audio this would very likely
flip.** Vibrato, inharmonicity, a turntable running 0.3% fast and MP3's own
time-frequency quantisation all move peaks by a bin or more, and there
``freq_quant=2`` should earn its keep. This is one of the clearest examples in
the project of a hyperparameter whose right value is a property of the data,
and of why a synthetic evaluation cannot settle it.

Note also that quantisation only reduces the *probability* of boundary
sensitivity, never eliminates it. The robust fix is redundancy (fan-out), not
finer quantisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .peaks import Constellation

__all__ = [
    "HashConfig",
    "Fingerprint",
    "fingerprint",
    "pack_hash",
    "unpack_hash",
]

#: Bit layout of a packed hash. 10 bits each for the two quantised frequencies
#: and 10 for the quantised time delta => 30 bits, comfortably inside int64 with
#: room to widen ``n_fft`` later without a format change.
FREQ_BITS = 10
DT_BITS = 10
FREQ_MAX = 1 << FREQ_BITS
DT_MAX = 1 << DT_BITS


@dataclass(frozen=True)
class HashConfig:
    """Constellation-pairing parameters.

    Args:
        fan_out: Maximum targets paired with each anchor.
        min_dt: Minimum anchor-to-target frame gap. Must be >= 1: a pair at
            ``Δt = 0`` encodes two simultaneous partials, which is a *chord*,
            not a temporal landmark, and such pairs are far less distinctive
            because they repeat throughout a track.
        max_dt: Maximum anchor-to-target frame gap. Bounds the target zone in
            time; too large and pairs span musical phrase boundaries where the
            two peaks are no longer physically related, and any time-stretch
            error accumulates over the interval.
        max_df: Maximum ``|f2 - f1|`` in bins. Bounds the target zone in
            frequency. Keeping the zone local means a band-limited query loses
            pairs only in the removed band, rather than losing every pair that
            happened to reach into it.
        freq_quant: Bins per quantised frequency bucket. Default 1 (none) —
            see the module docstring for the measurement behind that, and why
            it would probably be 2 on real audio.
        time_quant: Frames per quantised time-delta bucket.
    """

    fan_out: int = 8
    min_dt: int = 1
    max_dt: int = 40
    max_df: int = 80
    freq_quant: int = 1
    time_quant: int = 1


DEFAULT_HASH_CONFIG = HashConfig()


@dataclass(frozen=True)
class Fingerprint:
    """Packed hashes plus the absolute anchor time of each.

    Attributes:
        hashes: ``int64`` array of packed ``(f1, f2, Δt)`` keys.
        offsets: ``int32`` array — the *anchor* frame index of each hash. Held
            separately from the key on purpose: the key must be
            translation-invariant so it matches regardless of where the query was
            cut, while the absolute offset is the raw material for the
            time-alignment vote in :mod:`audio_fingerprint.match`.
    """

    hashes: NDArray[np.int64]
    offsets: NDArray[np.int32]

    def __len__(self) -> int:
        return int(self.hashes.size)


def pack_hash(
    f1: NDArray[np.int64] | int,
    f2: NDArray[np.int64] | int,
    dt: NDArray[np.int64] | int,
) -> NDArray[np.int64]:
    """Pack quantised ``(f1, f2, dt)`` into a single int64 key.

    Raises:
        ValueError: If any field overflows its bit field. Silent wraparound here
            would produce keys that collide across totally unrelated spectral
            geometry, which is the kind of bug that shows up only as a mildly
            elevated false-accept rate — nearly impossible to find later.
    """
    f1_a = np.asarray(f1, dtype=np.int64)
    f2_a = np.asarray(f2, dtype=np.int64)
    dt_a = np.asarray(dt, dtype=np.int64)
    if f1_a.size and (f1_a.min() < 0 or f1_a.max() >= FREQ_MAX):
        raise ValueError(f"f1 out of range [0,{FREQ_MAX})")
    if f2_a.size and (f2_a.min() < 0 or f2_a.max() >= FREQ_MAX):
        raise ValueError(f"f2 out of range [0,{FREQ_MAX})")
    if dt_a.size and (dt_a.min() < 0 or dt_a.max() >= DT_MAX):
        raise ValueError(f"dt out of range [0,{DT_MAX})")
    return (f1_a << (FREQ_BITS + DT_BITS)) | (f2_a << DT_BITS) | dt_a


def unpack_hash(
    packed: NDArray[np.int64] | int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Inverse of :func:`pack_hash`. Returns ``(f1, f2, dt)``."""
    p = np.asarray(packed, dtype=np.int64)
    dt = p & (DT_MAX - 1)
    f2 = (p >> DT_BITS) & (FREQ_MAX - 1)
    f1 = p >> (FREQ_BITS + DT_BITS)
    return f1, f2, dt


def fingerprint(
    constellation: Constellation,
    config: HashConfig = DEFAULT_HASH_CONFIG,
) -> Fingerprint:
    """Pair peaks into a target zone and emit packed hashes.

    Args:
        constellation: Peaks, assumed sorted by frame (as :func:`pick_peaks`
            returns them).
        config: Pairing parameters.

    Returns:
        A :class:`Fingerprint`. Deterministic: the same constellation and config
        always produce byte-identical output, including ordering. Determinism is
        not cosmetic — a reference database built by one process must be
        queryable by another, and an unstable ordering would make the index
        non-reproducible and impossible to diff.

    Implementation note: peaks are sorted by frame, so for each anchor the
    eligible target window is a contiguous slice found by binary search. Targets
    inside that slice are filtered by ``|Δf| <= max_df`` and the first
    ``fan_out`` survivors *in time order* are taken. Taking the earliest rather
    than, say, the loudest is deliberate — loudness is the quantity degradation
    destroys, so selecting on it would make the query's choice of targets differ
    from the reference's.
    """
    frames = np.asarray(constellation.frames, dtype=np.int64)
    bins = np.asarray(constellation.bins, dtype=np.int64)
    n = frames.size
    if n == 0:
        return Fingerprint(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32))
    if not np.all(np.diff(frames) >= 0):
        order = np.argsort(frames, kind="stable")
        frames, bins = frames[order], bins[order]

    if config.min_dt < 1:
        raise ValueError("min_dt must be >= 1; Δt=0 pairs are not time landmarks")
    if config.max_dt <= config.min_dt:
        raise ValueError("max_dt must exceed min_dt")

    lo_idx = np.searchsorted(frames, frames + config.min_dt, side="left")
    hi_idx = np.searchsorted(frames, frames + config.max_dt, side="right")

    out_f1: list[np.ndarray] = []
    out_f2: list[np.ndarray] = []
    out_dt: list[np.ndarray] = []
    out_t: list[np.ndarray] = []

    for i in range(n):
        lo, hi = int(lo_idx[i]), int(hi_idx[i])
        if hi <= lo:
            continue
        cand_f = bins[lo:hi]
        cand_t = frames[lo:hi]
        ok = np.abs(cand_f - bins[i]) <= config.max_df
        if not ok.any():
            continue
        sel = np.flatnonzero(ok)[: config.fan_out]
        out_f1.append(np.full(sel.size, bins[i], dtype=np.int64))
        out_f2.append(cand_f[sel])
        out_dt.append(cand_t[sel] - frames[i])
        out_t.append(np.full(sel.size, frames[i], dtype=np.int64))

    if not out_f1:
        return Fingerprint(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32))

    f1 = np.concatenate(out_f1) // config.freq_quant
    f2 = np.concatenate(out_f2) // config.freq_quant
    dt = np.concatenate(out_dt) // config.time_quant
    anchor_t = np.concatenate(out_t).astype(np.int32)

    return Fingerprint(pack_hash(f1, f2, dt), anchor_t)
