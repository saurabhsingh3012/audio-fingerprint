"""Inverted index: hash key -> postings of ``(track_id, anchor_offset)``.

Why an inverted index rather than comparing fingerprints pairwise
----------------------------------------------------------------
The naive approach — compute the query fingerprint, then compare it against
every reference fingerprint in turn — is ``O(corpus_size)`` per query. Fine for
60 synthetic tracks, hopeless for anything real. The inverted index turns
retrieval into ``O(query_hashes x mean_posting_length)``, which is roughly
independent of the number of tracks *provided the hash space stays sparse*. That
proviso is why the hash design in :mod:`audio_fingerprint.hashing` matters so
much: a low-entropy key gives long posting lists and the index degenerates back
towards a linear scan.

Storage layout, and why it is CSR rather than a dict of lists
-------------------------------------------------------------
The index is held as four flat arrays — a sorted key array plus per-key counts
and the concatenated ``track_id`` / ``offset`` postings — i.e. the same layout a
sparse CSR matrix uses. A ``dict[int, list[tuple[int, int]]]`` is the obvious
implementation and is the wrong one here for a specific reason: querying it
requires a Python-level loop over every query hash, and a 10-second query has
several thousand hashes. The flat layout lets a whole query be resolved with one
vectorised ``searchsorted`` plus one ragged gather, which is what makes the
1,000-query evaluation grid run in seconds instead of minutes.

It also serialises directly, without pickle. Pickle would be shorter and would
(a) be unsafe to load from an untrusted source and (b) tie the on-disk format to
the Python class layout, so a refactor silently invalidates every stored index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .hashing import Fingerprint

__all__ = ["FingerprintIndex", "IndexStats"]


@dataclass(frozen=True)
class IndexStats:
    """Summary statistics used for sizing and sanity checks."""

    n_tracks: int
    n_distinct_hashes: int
    n_postings: int
    mean_posting_length: float
    max_posting_length: int
    hash_collision_ratio: float
    """``n_postings / n_distinct_hashes`` — how much the key space is being
    reused. Close to 1.0 means keys are highly specific; a large value means
    many unrelated moments in the corpus are producing the same key, which is
    the early warning sign that the hash is too coarse for the corpus size."""


@dataclass
class FingerprintIndex:
    """In-memory inverted index over constellation hashes.

    Build with :meth:`add_track`, then query with :meth:`lookup_many`. The
    transition is handled by :meth:`freeze`, called lazily.
    """

    track_names: dict[int, str] = field(default_factory=dict)
    _blocks: list[tuple[int, NDArray[np.int64], NDArray[np.int32]]] = field(
        default_factory=list, repr=False
    )
    _keys: NDArray[np.int64] | None = field(default=None, repr=False)
    _starts: NDArray[np.int64] | None = field(default=None, repr=False)
    _counts: NDArray[np.int64] | None = field(default=None, repr=False)
    _post_tracks: NDArray[np.int32] | None = field(default=None, repr=False)
    _post_offsets: NDArray[np.int32] | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- building

    def add_track(self, track_id: int, name: str, fp: Fingerprint) -> None:
        """Insert every hash of ``fp`` under ``track_id``.

        Args:
            track_id: Stable integer identifier.
            name: Display name.
            fp: Fingerprint of the *full reference* recording.

        Raises:
            ValueError: If ``track_id`` is already present. Re-adding a track
                would double its postings and quietly bias every future vote
                towards it — a nasty class of bug to chase, because it degrades
                accuracy without ever raising an error.
        """
        if track_id in self.track_names:
            raise ValueError(f"track_id {track_id} already indexed")
        self.track_names[track_id] = name
        self._blocks.append(
            (int(track_id), np.asarray(fp.hashes, dtype=np.int64),
             np.asarray(fp.offsets, dtype=np.int32))
        )
        self._keys = None  # invalidate

    @property
    def is_frozen(self) -> bool:
        """True if the CSR arrays are built and the index is query-ready."""
        return self._keys is not None

    def freeze(self) -> None:
        """Build the CSR arrays. Idempotent; called lazily by queries."""
        if self._keys is not None:
            return
        if not self._blocks:
            self._keys = np.zeros(0, dtype=np.int64)
            self._starts = np.zeros(0, dtype=np.int64)
            self._counts = np.zeros(0, dtype=np.int64)
            self._post_tracks = np.zeros(0, dtype=np.int32)
            self._post_offsets = np.zeros(0, dtype=np.int32)
            return

        all_hashes = np.concatenate([h for _, h, _ in self._blocks])
        all_tracks = np.concatenate(
            [np.full(h.size, tid, dtype=np.int32) for tid, h, _ in self._blocks]
        )
        all_offsets = np.concatenate([o for _, _, o in self._blocks])

        # Stable sort so the posting order for a given key is deterministic
        # (insertion order within a track, track order by insertion). Index
        # files must be byte-reproducible for the round-trip test to mean
        # anything.
        order = np.argsort(all_hashes, kind="stable")
        sorted_hashes = all_hashes[order]
        self._post_tracks = np.ascontiguousarray(all_tracks[order])
        self._post_offsets = np.ascontiguousarray(all_offsets[order])

        keys, starts, counts = np.unique(
            sorted_hashes, return_index=True, return_counts=True
        )
        self._keys = np.ascontiguousarray(keys.astype(np.int64))
        self._starts = np.ascontiguousarray(starts.astype(np.int64))
        self._counts = np.ascontiguousarray(counts.astype(np.int64))

    # ---------------------------------------------------------------- querying

    def lookup(self, hash_key: int) -> tuple[NDArray[np.int32], NDArray[np.int32]] | None:
        """Return ``(track_ids, offsets)`` for one key, or ``None`` if absent."""
        self.freeze()
        assert self._keys is not None
        pos = int(np.searchsorted(self._keys, np.int64(hash_key)))
        if pos >= self._keys.size or int(self._keys[pos]) != int(hash_key):
            return None
        assert self._starts is not None and self._counts is not None
        assert self._post_tracks is not None and self._post_offsets is not None
        lo = int(self._starts[pos])
        hi = lo + int(self._counts[pos])
        return self._post_tracks[lo:hi], self._post_offsets[lo:hi]

    def lookup_many(
        self,
        hash_keys: NDArray[np.int64],
        query_offsets: NDArray[np.int32],
        *,
        max_posting_length: int | None = None,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Resolve a whole query at once.

        Args:
            hash_keys: Query hash keys.
            query_offsets: Anchor frame of each query hash, same length.
            max_posting_length: Skip keys whose posting list is longer than
                this. A key shared by a large fraction of the corpus carries
                almost no information — the same reasoning as a stop word in
                text retrieval — while costing the most to process.

        Returns:
            ``(track_ids, deltas)`` flat arrays, where
            ``delta = reference_offset - query_offset`` for every collision.
            These are the raw votes; :mod:`audio_fingerprint.match` histograms
            them.
        """
        self.freeze()
        assert self._keys is not None and self._starts is not None
        assert self._counts is not None
        assert self._post_tracks is not None and self._post_offsets is not None

        q_hashes = np.asarray(hash_keys, dtype=np.int64)
        q_offsets = np.asarray(query_offsets, dtype=np.int32)
        if q_hashes.size == 0 or self._keys.size == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

        pos = np.searchsorted(self._keys, q_hashes)
        pos_clipped = np.minimum(pos, self._keys.size - 1)
        hit = self._keys[pos_clipped] == q_hashes
        if max_posting_length is not None:
            hit &= self._counts[pos_clipped] <= max_posting_length
        if not hit.any():
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

        sel = pos_clipped[hit]
        starts = self._starts[sel]
        counts = self._counts[sel]
        total = int(counts.sum())

        # Ragged gather: expand each (start, count) run into flat indices
        # without a Python loop.
        run_start = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        flat_idx = np.repeat(starts, counts) + (np.arange(total) - run_start)

        track_ids = self._post_tracks[flat_idx]
        deltas = self._post_offsets[flat_idx].astype(np.int32) - np.repeat(
            q_offsets[hit], counts
        ).astype(np.int32)
        return track_ids, deltas

    def __contains__(self, hash_key: int) -> bool:
        return self.lookup(hash_key) is not None

    def __len__(self) -> int:
        """Number of distinct hash keys."""
        self.freeze()
        assert self._keys is not None
        return int(self._keys.size)

    # ---------------------------------------------------------------- reporting

    def stats(self) -> IndexStats:
        """Compute :class:`IndexStats` for the current contents."""
        self.freeze()
        assert self._counts is not None and self._keys is not None
        if self._counts.size == 0:
            return IndexStats(len(self.track_names), 0, 0, 0.0, 0, 0.0)
        n_postings = int(self._counts.sum())
        n_keys = int(self._keys.size)
        return IndexStats(
            n_tracks=len(self.track_names),
            n_distinct_hashes=n_keys,
            n_postings=n_postings,
            mean_posting_length=float(self._counts.mean()),
            max_posting_length=int(self._counts.max()),
            hash_collision_ratio=n_postings / n_keys,
        )

    # ---------------------------------------------------------------- storage

    def save(self, path: str | Path) -> None:
        """Serialise to a compressed ``.npz``."""
        self.freeze()
        assert self._keys is not None and self._counts is not None
        assert self._post_tracks is not None and self._post_offsets is not None
        name_ids = np.asarray(sorted(self.track_names), dtype=np.int64)
        names = np.asarray([self.track_names[int(i)] for i in name_ids], dtype="<U256")
        np.savez_compressed(
            Path(path),
            keys=self._keys,
            counts=self._counts,
            post_tracks=self._post_tracks,
            post_offsets=self._post_offsets,
            name_ids=name_ids,
            names=names,
        )

    @classmethod
    def load(cls, path: str | Path) -> FingerprintIndex:
        """Load an index written by :meth:`save`.

        The result is query-equivalent and array-identical to the saved index;
        ``tests/test_index.py`` asserts the round trip element by element.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            keys = np.ascontiguousarray(data["keys"].astype(np.int64))
            counts = np.ascontiguousarray(data["counts"].astype(np.int64))
            post_tracks = np.ascontiguousarray(data["post_tracks"].astype(np.int32))
            post_offsets = np.ascontiguousarray(data["post_offsets"].astype(np.int32))
            name_ids = data["name_ids"]
            names = data["names"]

        idx = cls()
        idx.track_names = {
            int(i): str(n) for i, n in zip(name_ids.tolist(), names.tolist(), strict=True)
        }
        idx._keys = keys
        idx._counts = counts
        idx._starts = (
            np.zeros(0, dtype=np.int64)
            if counts.size == 0
            else np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        )
        idx._post_tracks = post_tracks
        idx._post_offsets = post_offsets
        return idx
