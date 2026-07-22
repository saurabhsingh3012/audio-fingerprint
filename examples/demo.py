"""Runnable demo: identify a degraded excerpt, and show why offset voting wins.

    The corpus is SYNTHETIC (procedurally generated tones, not music).

Run:  python examples/demo.py

It does two things, both on real (not hard-coded) pipeline output:

1. Identifies a short, noisy excerpt of one track against a database of others,
   printing the recovered track id, alignment offset, and score.
2. Runs the head-to-head that motivates the whole matcher: offset-histogram
   voting versus a raw-hash-count baseline, over a batch of hard queries. Voting
   should win clearly. This is the number the README quotes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_fingerprint import DEFAULT_CONFIG as CFG  # noqa: E402
from audio_fingerprint import fingerprint_audio, synth  # noqa: E402
from audio_fingerprint.index import FingerprintIndex  # noqa: E402
from audio_fingerprint.match import identify, rank_by_raw_hash_count  # noqa: E402


def build(n_tracks: int = 48, seconds: float = 25.0, seed: int = 2024):
    corpus = synth.make_corpus(n_tracks, duration_s=seconds, seed=seed)
    index = FingerprintIndex()
    for spec in corpus.specs:
        index.add_track(
            spec.track_id, spec.name, fingerprint_audio(corpus.audio[spec.track_id], CFG)
        )
    index.freeze()
    return corpus, index


def demo_single_identification(corpus, index, rng) -> None:
    target = corpus.specs[7]
    excerpt, start_s = synth.crop(corpus.audio[target.track_id], CFG.sample_rate, 2.0, rng)
    noisy = synth.add_noise(excerpt, snr_db=0.0, rng=rng)  # 0 dB: as loud as the signal
    results = identify(index, fingerprint_audio(noisy, CFG), spec_config=CFG)

    print("Single-query identification")
    print(f"  true track     : {target.name} (id {target.track_id})")
    print(f"  query          : 2.0 s excerpt from {start_s:.2f}s, 0 dB SNR white noise")
    top = results[0]
    verdict = "CORRECT" if top.track_id == target.track_id else "WRONG"
    runner_up = f"{results[1].score:.0f}" if len(results) > 1 else "n/a"
    print(f"  identified as  : {top.name} (id {top.track_id})  {verdict}")
    print(f"  recovered start: {top.offset_seconds:.2f} s  (true {start_s:.2f} s)")
    print(f"  score          : {top.score:.0f}   runner-up: {runner_up}")


def demo_voting_vs_raw_count(corpus, index, rng) -> None:
    vote_ok = raw_ok = 0
    n = len(corpus.specs)
    for spec in corpus.specs:
        excerpt, _ = synth.crop(corpus.audio[spec.track_id], CFG.sample_rate, 2.0, rng)
        noisy = synth.add_noise(excerpt, 0.0, rng)
        fp = fingerprint_audio(noisy, CFG)
        voted = identify(index, fp)
        raw = rank_by_raw_hash_count(index, fp)
        vote_ok += bool(voted and voted[0].track_id == spec.track_id)
        raw_ok += bool(raw and raw[0].track_id == spec.track_id)
    print("\nOffset voting vs raw hash count  (2 s excerpts, 0 dB SNR)")
    print(f"  queries              : {n}")
    print(f"  offset-voting top-1  : {vote_ok / n:.3f}")
    print(f"  raw-count top-1      : {raw_ok / n:.3f}")
    print("  -> alignment structure, not overlap volume, is the signal.")


def main() -> None:
    print("=" * 66)
    print("audio-fingerprint demo  (SYNTHETIC corpus — not music)")
    print("=" * 66)
    corpus, index = build()
    print(f"indexed {len(corpus.specs)} tracks, {len(index):,} distinct hash keys\n")
    rng = np.random.default_rng(11)
    demo_single_identification(corpus, index, rng)
    demo_voting_vs_raw_count(corpus, index, rng)


if __name__ == "__main__":
    main()
