"""Matching: the load-bearing claim that offset voting beats raw hash counting.

The headline test constructs a near-miss where a decoy track shares *more*
hashes with the query than the correct track does, but shares them at
inconsistent time offsets. Offset-histogram voting must still pick the correct
track; the raw-count baseline must pick the decoy. If this ever fails, the
central design claim of the project is false.
"""

from __future__ import annotations

import numpy as np

from audio_fingerprint import DEFAULT_CONFIG, fingerprint_audio, synth
from audio_fingerprint.hashing import Fingerprint
from audio_fingerprint.index import FingerprintIndex
from audio_fingerprint.match import identify, rank_by_raw_hash_count


def test_offset_voting_beats_raw_count_on_constructed_near_miss():
    """Correct track: fewer hashes, all aligned. Decoy: more hashes, scattered.

    Construction (all synthetic hash keys, no audio needed so the test is
    exact and fast):

    * Query has 40 distinct keys at known anchor offsets.
    * Track CORRECT (id 0) contains 12 of those keys, each at
      ``query_offset + 100`` — so all 12 collisions land on delta = 100.
    * Track DECOY (id 1) contains 25 of those keys, each at a *different*
      contrived ref offset so their deltas are all distinct — no two votes
      share a bin.

    Raw hash count: decoy 25 > correct 12  -> decoy wins.
    Offset voting: correct's tallest bin = 12, decoy's tallest bin = 1
      -> correct wins by a wide margin.
    """
    rng = np.random.default_rng(0)
    query_keys = rng.choice(np.arange(1000, 5000), size=40, replace=False).astype(np.int64)
    query_offsets = np.arange(40, dtype=np.int32) * 3  # arbitrary, spread out
    query_fp = Fingerprint(query_keys, query_offsets)

    idx = FingerprintIndex()

    # CORRECT track: 12 keys, all aligned at delta = +100.
    correct_sel = np.arange(12)
    correct_fp = Fingerprint(
        hashes=query_keys[correct_sel],
        offsets=(query_offsets[correct_sel] + 100).astype(np.int32),
    )
    idx.add_track(0, "CORRECT", correct_fp)

    # DECOY track: 25 keys, each at a ref offset giving a unique delta.
    decoy_sel = np.arange(15, 40)  # 25 keys, disjoint-ish from correct
    decoy_offsets = (query_offsets[decoy_sel] + np.arange(25) * 37 + 500).astype(np.int32)
    idx.add_track(1, "DECOY", Fingerprint(query_keys[decoy_sel], decoy_offsets))
    idx.freeze()

    # Raw hash count ranks the decoy first.
    raw = rank_by_raw_hash_count(idx, query_fp)
    assert raw[0].track_id == 1, "decoy should win on raw hash count"
    assert raw[0].raw_hits > raw[1].raw_hits

    # Offset voting ranks the correct track first.
    voted = identify(idx, query_fp)
    assert voted[0].track_id == 0, "offset voting must recover the aligned track"
    assert voted[0].score == 12
    assert voted[0].offset_frames == 100
    # And it wins decisively over the decoy's best (scattered) bin.
    decoy_voted = next(r for r in voted if r.track_id == 1)
    assert voted[0].score > decoy_voted.score


def test_identify_recovers_true_offset_on_real_audio(small_corpus):
    """On real synthetic audio, the winning offset equals the crop position."""
    cfg = DEFAULT_CONFIG
    idx = FingerprintIndex()
    for spec in small_corpus.specs:
        idx.add_track(spec.track_id, spec.name,
                      fingerprint_audio(small_corpus.audio[spec.track_id], cfg))
    idx.freeze()

    rng = np.random.default_rng(7)
    target = small_corpus.specs[2].track_id
    excerpt, start_s = synth.crop(small_corpus.audio[target], cfg.sample_rate, 4.0, rng)
    excerpt = synth.add_noise(excerpt, 15.0, rng)

    res = identify(idx, fingerprint_audio(excerpt, cfg), spec_config=cfg)
    assert res[0].track_id == target
    # Recovered offset should be within ~150 ms of the true crop start.
    assert abs(res[0].offset_seconds - start_s) < 0.15


def test_empty_query_returns_no_results():
    idx = FingerprintIndex()
    idx.add_track(0, "a", Fingerprint(np.array([1], np.int64), np.array([0], np.int32)))
    idx.freeze()
    empty = Fingerprint(np.zeros(0, np.int64), np.zeros(0, np.int32))
    assert identify(idx, empty) == []


def test_out_of_corpus_query_scores_low(small_corpus):
    """A query from an unindexed track should not produce a confident match.

    This is the false-accept guard in miniature: build an index, then query it
    with audio it has never seen and assert the top score is far below what a
    genuine match produces. Uses generous margins so the test is not flaky, but
    the direction of the inequality is the point.
    """
    cfg = DEFAULT_CONFIG
    idx = FingerprintIndex()
    for spec in small_corpus.specs:
        idx.add_track(spec.track_id, spec.name,
                      fingerprint_audio(small_corpus.audio[spec.track_id], cfg))
    idx.freeze()

    impostor = synth.make_impostor_corpus(1, duration_s=8.0, seed=777)
    imp_audio = impostor.audio[impostor.specs[0].track_id]
    rng = np.random.default_rng(1)
    excerpt, _ = synth.crop(imp_audio, cfg.sample_rate, 4.0, rng)
    imp_res = identify(idx, fingerprint_audio(excerpt, cfg), spec_config=cfg)
    imp_score = imp_res[0].score if imp_res else 0

    genuine_audio = small_corpus.audio[small_corpus.specs[0].track_id]
    g_excerpt, _ = synth.crop(genuine_audio, cfg.sample_rate, 4.0, rng)
    g_res = identify(idx, fingerprint_audio(g_excerpt, cfg), spec_config=cfg)
    genuine_score = g_res[0].score

    assert genuine_score > 3 * max(imp_score, 1)
