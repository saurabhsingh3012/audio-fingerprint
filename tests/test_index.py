"""Index: build correctness, lookup, and save/load round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from audio_fingerprint import DEFAULT_CONFIG, fingerprint_audio
from audio_fingerprint.hashing import Fingerprint
from audio_fingerprint.index import FingerprintIndex


def _build(small_corpus) -> FingerprintIndex:
    idx = FingerprintIndex()
    for spec in small_corpus.specs:
        idx.add_track(
            spec.track_id, spec.name,
            fingerprint_audio(small_corpus.audio[spec.track_id], DEFAULT_CONFIG),
        )
    idx.freeze()
    return idx


def test_lookup_matches_inserted_postings():
    """Every (key, track, offset) inserted is retrievable via lookup."""
    idx = FingerprintIndex()
    fp = Fingerprint(
        hashes=np.array([10, 10, 20, 30], dtype=np.int64),
        offsets=np.array([1, 2, 3, 4], dtype=np.int32),
    )
    idx.add_track(0, "a", fp)
    idx.freeze()
    tracks, offsets = idx.lookup(10)
    assert set(offsets.tolist()) == {1, 2}
    assert set(tracks.tolist()) == {0}
    assert idx.lookup(999) is None
    assert 10 in idx
    assert 999 not in idx


def test_duplicate_track_id_rejected():
    idx = FingerprintIndex()
    fp = Fingerprint(np.array([1], dtype=np.int64), np.array([0], dtype=np.int32))
    idx.add_track(0, "a", fp)
    with pytest.raises(ValueError):
        idx.add_track(0, "b", fp)


def test_lookup_many_returns_correct_deltas():
    """lookup_many yields ref_offset - query_offset for each collision."""
    idx = FingerprintIndex()
    idx.add_track(0, "a", Fingerprint(
        np.array([5, 5, 7], dtype=np.int64),
        np.array([100, 200, 300], dtype=np.int32),
    ))
    idx.freeze()
    # Query has key 5 at offset 10 and key 7 at offset 50.
    tracks, deltas = idx.lookup_many(
        np.array([5, 7], dtype=np.int64), np.array([10, 50], dtype=np.int32)
    )
    # key 5 -> ref offsets {100,200} => deltas {90,190}; key 7 -> 300-50=250.
    by_track = sorted(zip(tracks.tolist(), deltas.tolist()))
    assert by_track == [(0, 90), (0, 190), (0, 250)]


def test_save_load_round_trip(small_corpus, tmp_path):
    """A saved index reloads array-identical and query-equivalent."""
    idx = _build(small_corpus)
    path = tmp_path / "idx.npz"
    idx.save(path)
    loaded = FingerprintIndex.load(path)

    assert loaded.track_names == idx.track_names
    np.testing.assert_array_equal(loaded._keys, idx._keys)
    np.testing.assert_array_equal(loaded._counts, idx._counts)
    np.testing.assert_array_equal(loaded._post_tracks, idx._post_tracks)
    np.testing.assert_array_equal(loaded._post_offsets, idx._post_offsets)

    # Query equivalence on a real fingerprint.
    fp = fingerprint_audio(
        small_corpus.audio[small_corpus.specs[0].track_id], DEFAULT_CONFIG
    )
    t1, d1 = idx.lookup_many(fp.hashes, fp.offsets)
    t2, d2 = loaded.lookup_many(fp.hashes, fp.offsets)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(d1, d2)


def test_stats_are_consistent(small_corpus):
    idx = _build(small_corpus)
    st = idx.stats()
    assert st.n_tracks == len(small_corpus.specs)
    assert st.n_distinct_hashes == len(idx)
    assert st.n_postings >= st.n_distinct_hashes
    assert st.mean_posting_length == pytest.approx(
        st.n_postings / st.n_distinct_hashes
    )


def test_empty_index_round_trip(tmp_path):
    idx = FingerprintIndex()
    idx.freeze()
    assert len(idx) == 0
    path = tmp_path / "empty.npz"
    idx.save(path)
    loaded = FingerprintIndex.load(path)
    assert len(loaded) == 0
