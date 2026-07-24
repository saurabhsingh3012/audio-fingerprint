"""End-to-end retrieval evaluation on a corpus of REAL Creative-Commons music.

    ***********************************************************************
    * THE CORPUS HERE IS REAL RECORDED MUSIC (not synthetic tones).       *
    * Tracks are Creative-Commons audio from the Internet Archive          *
    * netlabels collection; see src/audio_fingerprint/corpus.py and the    *
    * committed DATA_SOURCES.md. Numbers this script prints are measured    *
    * on that audio and are the honest counterpart to the SYNTHETIC        *
    * upper bound produced by tests/evaluate.py.                           *
    ***********************************************************************

What this measures, and how it relates to the synthetic evaluation
------------------------------------------------------------------
This is the *same* pipeline and the *same* degradation axes as
``tests/evaluate.py`` — SNR, excerpt length, band-limit, an impostor / false-
accept analysis — run on real music instead of procedurally generated tones, so
the two are directly comparable. Holding the degradation constant and swapping
synthetic tones for real music isolates the one question the synthetic evaluation
could not answer: **how much does real music cost the algorithm?**

It adds two things the synthetic harness structurally could not test:

1. **A codec round-trip axis.** Each query is optionally re-encoded to MP3 via
   ``ffmpeg`` at a controlled bitrate and decoded back. Codecs quantise in exactly
   the time-frequency domain the fingerprint lives in — the single most important
   real-world degradation the synthetic evaluation was missing.
2. **A ``freq_quant`` 1-vs-2 comparison at a hard operating point.** The design
   doc predicts that turning frequency quantisation *on* (``freq_quant=2``), which
   did **not** help on stationary synthetic partials, *should* help on real audio
   where vibrato, inharmonicity and codec jitter move peaks by a bin. This run
   tests that prediction directly.

Run it (after installing the real extra: ``pip install -e ".[real]"``)::

    python tests/evaluate_real.py --json-out results/eval_real.json

The first run downloads ~48 CC tracks (~150 MB) into ``data/`` (git-ignored);
subsequent runs are offline. This script is deliberately **not** collected by
pytest and **not** in CI — CI runs the synthetic ``--quick`` path, which needs no
network. Nothing here is fabricated: every number comes from a run of the actual
pipeline on the actual audio.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_fingerprint import synth  # noqa: E402
from audio_fingerprint.corpus import RealCorpus, ensure_corpus  # noqa: E402
from audio_fingerprint.hashing import DEFAULT_HASH_CONFIG, HashConfig, fingerprint  # noqa: E402
from audio_fingerprint.index import FingerprintIndex  # noqa: E402
from audio_fingerprint.match import DEFAULT_MATCH_CONFIG, identify  # noqa: E402
from audio_fingerprint.peaks import DEFAULT_PEAK_CONFIG, pick_peaks  # noqa: E402
from audio_fingerprint.spectrogram import DEFAULT_CONFIG  # noqa: E402

CFG = DEFAULT_CONFIG
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------
# Fingerprinting + codec degradation
# --------------------------------------------------------------------------


def _fp(audio, hash_config: HashConfig = DEFAULT_HASH_CONFIG):
    """Audio -> fingerprint at the given hash configuration."""
    return fingerprint(pick_peaks(audio, CFG, DEFAULT_PEAK_CONFIG), hash_config)


def mp3_roundtrip(x: NDArray[np.float64], sample_rate: int, bitrate: str) -> NDArray[np.float64]:
    """Encode ``x`` to MP3 at ``bitrate`` via ffmpeg and decode it back.

    This is a genuine lossy codec round-trip: it quantises the signal in the
    modified-DCT time-frequency domain, which is precisely where the fingerprint's
    peaks live, so it is the most fingerprint-relevant degradation in the suite.
    The encoder/decoder delay shifts the signal in time, but the offset-histogram
    vote is translation-invariant, so a constant shift costs nothing.
    """
    import librosa
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "in.wav")
        mp3 = os.path.join(d, "out.mp3")
        sf.write(wav, x, sample_rate)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", bitrate, mp3],
            check=True,
        )
        y, _ = librosa.load(mp3, sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float64)


def _degrade(
    clean: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    excerpt_s: float,
    snr_db: float | None,
    band: tuple[float | None, float | None] | None,
    codec_bitrate: str | None = None,
) -> NDArray[np.float64]:
    """Crop -> band-limit -> add noise -> optional MP3 round-trip, in that order.

    The codec goes *last* so it acts on the fully degraded signal, mirroring a
    real query captured, band-limited by a microphone, buried in noise, and then
    compressed for transmission.
    """
    q, _ = synth.crop(clean, CFG.sample_rate, excerpt_s, rng)
    if band is not None:
        q = synth.band_limit(q, CFG.sample_rate, band[0], band[1])
    if snr_db is not None:
        q = synth.add_noise(q, snr_db, rng)
    if codec_bitrate is not None:
        q = mp3_roundtrip(q, CFG.sample_rate, codec_bitrate)
    return q


# --------------------------------------------------------------------------
# Metrics containers (mirrors tests/evaluate.py so the tables line up)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """Metrics for one grid cell."""

    axis: str
    label: str
    n_queries: int
    top1_accuracy: float
    mrr: float
    median_score: float
    median_true_score: float

    def row(self) -> str:
        return (
            f"| {self.label:<18} | {self.n_queries:>3} | "
            f"{self.top1_accuracy:>7.3f} | {self.mrr:>5.3f} | {self.median_score:>7.0f} |"
        )


@dataclass
class EvalReport:
    """Everything the run produced, for JSON serialisation and the README."""

    corpus: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    index_stats: dict = field(default_factory=dict)
    snr_sweep: list[CellResult] = field(default_factory=list)
    excerpt_sweep: list[CellResult] = field(default_factory=list)
    bandlimit_sweep: list[CellResult] = field(default_factory=list)
    codec_sweep: list[CellResult] = field(default_factory=list)
    freq_quant_comparison: list[dict] = field(default_factory=list)
    threshold_table: list[dict] = field(default_factory=list)
    operating_point: dict = field(default_factory=dict)
    impostor_summary: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0


# --------------------------------------------------------------------------
# Query / cell drivers
# --------------------------------------------------------------------------


def _query_once(
    index: FingerprintIndex, audio, true_id: int,
    hash_config: HashConfig = DEFAULT_HASH_CONFIG,
) -> tuple[int, float, float]:
    """Run one query; return ``(rank_of_true, top_score, true_track_score)``.

    ``hash_config`` MUST match the config the index was built with — a reference
    and a query are only comparable if fingerprinted identically.
    """
    results = identify(index, _fp(audio, hash_config), DEFAULT_MATCH_CONFIG, CFG)
    if not results:
        return 0, 0.0, 0.0
    top_score = results[0].score
    for i, r in enumerate(results, start=1):
        if r.track_id == true_id:
            return i, top_score, r.score
    return 0, top_score, 0.0


def _run_cell(
    index: FingerprintIndex,
    corpus: RealCorpus,
    rng: np.random.Generator,
    axis: str,
    label: str,
    *,
    excerpt_s: float,
    snr_db: float | None,
    band: tuple[float | None, float | None] | None,
    codec_bitrate: str | None,
    queries_per_track: int,
) -> tuple[CellResult, list[tuple[int, float, float]]]:
    """Evaluate one grid cell over the reference tracks; return metrics + records."""
    records: list[tuple[int, float, float]] = []
    for spec in corpus.specs:
        for _ in range(queries_per_track):
            q = _degrade(
                corpus.audio[spec.track_id], rng, excerpt_s=excerpt_s,
                snr_db=snr_db, band=band, codec_bitrate=codec_bitrate,
            )
            records.append(_query_once(index, q, spec.track_id))
    ranks = np.asarray([r for r, _, _ in records])
    top_scores = np.asarray([s for _, s, _ in records])
    true_scores = np.asarray([t for _, _, t in records])
    cell = CellResult(
        axis=axis, label=label, n_queries=len(records),
        top1_accuracy=float(np.mean(ranks == 1)),
        mrr=float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0))),
        median_score=float(np.median(top_scores)),
        median_true_score=float(np.median(true_scores)),
    )
    return cell, records


def build_index(
    corpus: RealCorpus, hash_config: HashConfig = DEFAULT_HASH_CONFIG
) -> tuple[FingerprintIndex, float]:
    """Fingerprint and index every reference track; return the index and build time."""
    index = FingerprintIndex()
    t0 = time.perf_counter()
    for spec in corpus.specs:
        index.add_track(spec.track_id, spec.name, _fp(corpus.audio[spec.track_id], hash_config))
    index.freeze()
    return index, time.perf_counter() - t0


def _threshold_analysis(
    genuine: list[tuple[int, float, float]], impostor_scores: list[float]
) -> tuple[list[dict], dict]:
    """Sweep the accept threshold; report TAR and FAR at each. See tests/evaluate.py."""
    gen_correct = np.asarray([top for rank, top, _ in genuine if rank == 1], dtype=np.float64)
    n_gen = len(genuine)
    imp = np.asarray(impostor_scores, dtype=np.float64)
    table: list[dict] = []
    for thr in range(0, 61, 2):
        tar = float(np.sum(gen_correct >= thr)) / max(n_gen, 1)
        far = float(np.mean(imp >= thr)) if imp.size else 0.0
        table.append(dict(threshold=thr, true_accept_rate=round(tar, 4),
                          false_accept_rate=round(far, 4)))
    op: dict = {}
    for entry in table:
        if entry["false_accept_rate"] <= 0.01:
            op = dict(entry, policy="lowest threshold with FAR <= 1%")
            break
    if not op and table:
        op = dict(table[-1], policy="FAR never reached 1%; reporting strictest threshold")
    return table, op


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> EvalReport:
    """Run the full REAL-audio evaluation and print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reference", type=int, default=36, help="tracks indexed")
    parser.add_argument("--n-impostors", type=int, default=12, help="held-out tracks")
    parser.add_argument("--queries-per-track", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--no-download", action="store_true",
                        help="fail instead of downloading if the corpus is not cached")
    parser.add_argument("--quick", action="store_true",
                        help="small fast configuration (fewer cells); still needs a cached corpus")
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    print("=" * 74)
    print("audio-fingerprint — retrieval evaluation on a REAL Creative-Commons corpus")
    print("(the honest counterpart to the synthetic upper bound; see tests/evaluate.py)")
    print("=" * 74)
    print(f"loading corpus ({args.n_reference} reference + {args.n_impostors} impostor) ...")
    full = ensure_corpus(
        n_reference=args.n_reference, n_impostors=args.n_impostors,
        data_dir=args.data_dir, allow_download=not args.no_download,
    )
    reference = full.with_role("reference")
    impostors = full.with_role("impostor")
    total_audio_s = sum(len(a) for a in reference.audio.values()) / CFG.sample_rate
    print(f"reference: {len(reference)} tracks, {total_audio_s / 60:.1f} min of audio; "
          f"impostors: {len(impostors)} tracks")

    index, build_s = build_index(reference)
    stats = index.stats()
    print(f"indexed in {build_s:.1f}s: {stats.n_distinct_hashes:,} keys, "
          f"{stats.n_postings:,} postings, mean posting {stats.mean_posting_length:.2f}, "
          f"collision ratio {stats.hash_collision_ratio:.2f}")

    report = EvalReport(
        corpus=dict(
            source="archive.org netlabels (Creative Commons)",
            n_reference=len(reference), n_impostors=len(impostors),
            reference_audio_minutes=round(total_audio_s / 60, 1),
        ),
        config=dict(
            queries_per_track=args.queries_per_track, seed=args.seed, quick=args.quick,
            sample_rate=CFG.sample_rate, n_fft=CFG.n_fft, hop_length=CFG.hop_length,
            target_density=DEFAULT_PEAK_CONFIG.target_density,
            fan_out=DEFAULT_HASH_CONFIG.fan_out, freq_quant=DEFAULT_HASH_CONFIG.freq_quant,
        ),
        index_stats=asdict(stats),
    )

    rng = np.random.default_rng(args.seed + 1)
    pooled_genuine: list[tuple[int, float, float]] = []

    # --- Sweep 1: SNR (5 s excerpt, full band, no codec) — comparable to synthetic
    snr_grid = [20.0, 10.0, 5.0, 0.0, -5.0] if not args.quick else [10.0, 0.0]
    print("\nSNR sweep (5 s excerpt, full band, no codec):")
    for snr in snr_grid:
        cell, recs = _run_cell(index, reference, rng, "snr", f"{snr:+.0f} dB",
                               excerpt_s=5.0, snr_db=snr, band=None, codec_bitrate=None,
                               queries_per_track=args.queries_per_track)
        report.snr_sweep.append(cell)
        pooled_genuine.extend(recs)
        print("  " + cell.row())

    # --- Sweep 2: excerpt length (10 dB SNR, full band, no codec)
    exc_grid = [1.0, 2.0, 3.0, 5.0, 10.0] if not args.quick else [3.0, 10.0]
    print("\nExcerpt-length sweep (10 dB SNR, full band, no codec):")
    for exc in exc_grid:
        cell, recs = _run_cell(index, reference, rng, "excerpt", f"{exc:.0f} s",
                               excerpt_s=exc, snr_db=10.0, band=None, codec_bitrate=None,
                               queries_per_track=args.queries_per_track)
        report.excerpt_sweep.append(cell)
        pooled_genuine.extend(recs)
        print("  " + cell.row())

    # --- Sweep 3: band-limit (5 s excerpt, 15 dB SNR, no codec)
    bands: list[tuple[str, tuple[float | None, float | None] | None]] = [
        ("full band", None),
        ("telephone 300-3400", (300.0, 3400.0)),
        ("highpass 500", (500.0, None)),
        ("lowpass 2000", (None, 2000.0)),
    ]
    if args.quick:
        bands = bands[:2]
    print("\nBand-limit sweep (5 s excerpt, 15 dB SNR, no codec):")
    for label, band in bands:
        cell, recs = _run_cell(index, reference, rng, "band", label,
                               excerpt_s=5.0, snr_db=15.0, band=band, codec_bitrate=None,
                               queries_per_track=args.queries_per_track)
        report.bandlimit_sweep.append(cell)
        pooled_genuine.extend(recs)
        print("  " + cell.row())

    # --- Sweep 4: MP3 codec round-trip (5 s excerpt, 10 dB SNR, full band) — NEW
    codec_grid: list[tuple[str, str | None]] = [
        ("no codec", None), ("mp3 128k", "128k"), ("mp3 64k", "64k"),
        ("mp3 48k", "48k"), ("mp3 32k", "32k"),
    ]
    if args.quick:
        codec_grid = [("no codec", None), ("mp3 64k", "64k")]
    print("\nMP3 codec round-trip sweep (5 s excerpt, 10 dB SNR, full band):")
    for label, br in codec_grid:
        cell, recs = _run_cell(index, reference, rng, "codec", label,
                               excerpt_s=5.0, snr_db=10.0, band=None, codec_bitrate=br,
                               queries_per_track=args.queries_per_track)
        report.codec_sweep.append(cell)
        pooled_genuine.extend(recs)
        print("  " + cell.row())

    # --- Impostors: out-of-corpus tracks, mixed degradation including codec ----
    print("\nImpostor queries (out-of-corpus; mixed degradation incl. MP3 64k):")
    imp_conditions: list[tuple[float, float, str | None]] = (
        [(5.0, 10.0, "64k"), (3.0, 5.0, None), (5.0, 0.0, "64k")]
        if not args.quick else [(5.0, 5.0, "64k")]
    )
    imp_scores: list[float] = []
    for exc, snr, br in imp_conditions:
        for spec in impostors.specs:
            q = _degrade(impostors.audio[spec.track_id], rng,
                         excerpt_s=exc, snr_db=snr, band=None, codec_bitrate=br)
            results = identify(index, _fp(q), DEFAULT_MATCH_CONFIG, CFG)
            imp_scores.append(results[0].score if results else 0.0)
    imp_arr = np.asarray(imp_scores, dtype=np.float64)
    report.impostor_summary = dict(
        n_impostor_queries=int(imp_arr.size),
        max_score=float(imp_arr.max()) if imp_arr.size else 0.0,
        p95_score=float(np.percentile(imp_arr, 95)) if imp_arr.size else 0.0,
        median_score=float(np.median(imp_arr)) if imp_arr.size else 0.0,
    )
    isum = report.impostor_summary
    print(f"  {imp_arr.size} impostor queries: median {isum['median_score']:.0f}, "
          f"p95 {isum['p95_score']:.0f}, max {isum['max_score']:.0f}")

    # --- Threshold / false-accept analysis ------------------------------------
    table, op = _threshold_analysis(pooled_genuine, imp_scores)
    report.threshold_table = table
    report.operating_point = op
    print("\nThreshold analysis (genuine queries pooled across all sweeps):")
    print("  thr | true-accept | false-accept")
    for e in table:
        if e["threshold"] % 4 == 0:
            print(f"  {e['threshold']:>3} |   {e['true_accept_rate']:>6.3f}    |   "
                  f"{e['false_accept_rate']:>6.3f}")
    if op:
        print(f"\nchosen operating point: threshold={op['threshold']} -> "
              f"true-accept {op['true_accept_rate']:.3f}, "
              f"false-accept {op['false_accept_rate']:.3f}  ({op['policy']})")

    # --- freq_quant 1 vs 2 at a hard operating point (tests the design-doc claim)
    print("\nfreq_quant 1-vs-2 at a hard cell (3 s excerpt, 0 dB SNR, MP3 64k):")
    for fq in (1, 2):
        hc = HashConfig(fan_out=DEFAULT_HASH_CONFIG.fan_out, min_dt=DEFAULT_HASH_CONFIG.min_dt,
                        max_dt=DEFAULT_HASH_CONFIG.max_dt, max_df=DEFAULT_HASH_CONFIG.max_df,
                        freq_quant=fq, time_quant=DEFAULT_HASH_CONFIG.time_quant)
        idx_fq, _ = build_index(reference, hc)
        rng_fq = np.random.default_rng(args.seed + 7)
        ranks = []
        for spec in reference.specs:
            for _ in range(args.queries_per_track):
                q = _degrade(reference.audio[spec.track_id], rng_fq,
                             excerpt_s=3.0, snr_db=0.0, band=None, codec_bitrate="64k")
                # query fingerprinted with the SAME hash config as this index
                r, _, _ = _query_once(idx_fq, q, spec.track_id, hc)
                ranks.append(r)
        ranks_a = np.asarray(ranks)
        entry = dict(freq_quant=fq, n_queries=int(ranks_a.size),
                     top1_accuracy=round(float(np.mean(ranks_a == 1)), 4),
                     n_distinct_hashes=int(idx_fq.stats().n_distinct_hashes))
        report.freq_quant_comparison.append(entry)
        print(f"  freq_quant={fq}: top-1 {entry['top1_accuracy']:.3f} "
              f"({entry['n_distinct_hashes']:,} keys)")

    report.runtime_seconds = time.perf_counter() - t_start
    print(f"\ntotal runtime: {report.runtime_seconds:.1f}s")

    if args.json_out:
        out = dict(
            corpus=report.corpus, config=report.config, index_stats=report.index_stats,
            snr_sweep=[asdict(c) for c in report.snr_sweep],
            excerpt_sweep=[asdict(c) for c in report.excerpt_sweep],
            bandlimit_sweep=[asdict(c) for c in report.bandlimit_sweep],
            codec_sweep=[asdict(c) for c in report.codec_sweep],
            freq_quant_comparison=report.freq_quant_comparison,
            threshold_table=report.threshold_table,
            operating_point=report.operating_point,
            impostor_summary=report.impostor_summary,
            runtime_seconds=report.runtime_seconds,
        )
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}")

    return report


if __name__ == "__main__":
    main()
