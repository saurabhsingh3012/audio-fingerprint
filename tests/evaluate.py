"""End-to-end retrieval evaluation on a SYNTHETIC corpus.

    ***********************************************************************
    * THE CORPUS IS SYNTHETIC. Every number this script prints describes  *
    * retrieval of procedurally generated tone sequences, NOT music. It   *
    * is an UPPER BOUND on real-world behaviour. See src/.../synth.py.    *
    ***********************************************************************

What this measures, and why each piece is here
----------------------------------------------
Run with ``python tests/evaluate.py`` (add ``--quick`` for a small, fast run;
that is what CI uses). It:

1. Builds a reference index of ``--n-tracks`` synthetic tracks.
2. Issues degraded queries — cropped excerpts of indexed tracks — across three
   independent sweeps (SNR, excerpt length, band-limit), reporting **real
   measured** top-1 accuracy and mean reciprocal rank per cell.
3. Issues an **impostor** set: queries cropped from tracks that were *never
   indexed*. This is the only way to measure a false-accept rate. A system that
   is 98% accurate on in-corpus queries but confidently names a track for every
   out-of-corpus recording is useless, and impostors are what expose that.
4. Selects a score threshold and reports the false-accept rate and the genuine
   true-accept rate at it — the precision/recall trade the threshold controls.

Every metric is computed from a run of the actual pipeline. Nothing is assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Allow running as a bare script (python tests/evaluate.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_fingerprint import synth  # noqa: E402
from audio_fingerprint.hashing import DEFAULT_HASH_CONFIG, fingerprint  # noqa: E402
from audio_fingerprint.index import FingerprintIndex  # noqa: E402
from audio_fingerprint.match import DEFAULT_MATCH_CONFIG, identify  # noqa: E402
from audio_fingerprint.peaks import DEFAULT_PEAK_CONFIG, pick_peaks  # noqa: E402
from audio_fingerprint.spectrogram import DEFAULT_CONFIG  # noqa: E402

CFG = DEFAULT_CONFIG


@dataclass(frozen=True)
class CellResult:
    """Metrics for one grid cell."""

    axis: str
    label: str
    n_queries: int
    top1_accuracy: float
    mrr: float
    median_score: float
    median_true_score: float  # median score of the *correct* track, matched or not

    def row(self) -> str:
        return (
            f"| {self.label:<16} | {self.n_queries:>3} | "
            f"{self.top1_accuracy:>7.3f} | {self.mrr:>5.3f} | "
            f"{self.median_score:>7.0f} |"
        )


@dataclass
class EvalReport:
    """Everything the run produced, for JSON serialisation and the README."""

    config: dict = field(default_factory=dict)
    index_stats: dict = field(default_factory=dict)
    snr_sweep: list[CellResult] = field(default_factory=list)
    excerpt_sweep: list[CellResult] = field(default_factory=list)
    bandlimit_sweep: list[CellResult] = field(default_factory=list)
    threshold_table: list[dict] = field(default_factory=list)
    operating_point: dict = field(default_factory=dict)
    impostor_summary: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0


def _fp(audio: NDArray[np.float64]):
    """Audio -> fingerprint, at the default configuration."""
    return fingerprint(pick_peaks(audio, CFG, DEFAULT_PEAK_CONFIG), DEFAULT_HASH_CONFIG)


def _query_once(
    index: FingerprintIndex, audio: NDArray[np.float64], true_id: int
) -> tuple[int, float, float]:
    """Run one query. Returns ``(rank, top_score, true_track_score)``.

    ``rank`` is the 1-based position of the correct track (0 if not in the
    returned top-k). ``top_score`` is the score of the rank-1 candidate. The
    correct track's own score is reported separately so the threshold analysis
    can reason about genuine matches even when they are not ranked first.
    """
    fp = _fp(audio)
    results = identify(index, fp, DEFAULT_MATCH_CONFIG, CFG)
    if not results:
        return 0, 0.0, 0.0
    top_score = results[0].score
    rank = 0
    true_score = 0.0
    for i, r in enumerate(results, start=1):
        if r.track_id == true_id:
            rank = i
            true_score = r.score
            break
    return rank, top_score, true_score


def _degrade(
    clean: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    excerpt_s: float,
    snr_db: float | None,
    band: tuple[float | None, float | None] | None,
    gain_db: float = 0.0,
) -> NDArray[np.float64]:
    """Apply the standard degradation chain in a fixed order."""
    q, _ = synth.crop(clean, CFG.sample_rate, excerpt_s, rng)
    if band is not None:
        q = synth.band_limit(q, CFG.sample_rate, band[0], band[1])
    if snr_db is not None:
        q = synth.add_noise(q, snr_db, rng)
    if gain_db:
        q = synth.apply_gain(q, gain_db)
    return q


def _run_cell(
    index: FingerprintIndex,
    corpus: synth.Corpus,
    rng: np.random.Generator,
    axis: str,
    label: str,
    *,
    excerpt_s: float,
    snr_db: float | None,
    band: tuple[float | None, float | None] | None,
    queries_per_track: int,
) -> tuple[CellResult, list[tuple[int, float, float]]]:
    """Evaluate one grid cell; also return raw per-query records."""
    records: list[tuple[int, float, float]] = []
    ranks: list[int] = []
    top_scores: list[float] = []
    true_scores: list[float] = []
    for spec in corpus.specs:
        for _ in range(queries_per_track):
            q = _degrade(
                corpus.audio[spec.track_id], rng,
                excerpt_s=excerpt_s, snr_db=snr_db, band=band,
            )
            rank, top, true_s = _query_once(index, q, spec.track_id)
            records.append((rank, top, true_s))
            ranks.append(rank)
            top_scores.append(top)
            true_scores.append(true_s)
    ranks_a = np.asarray(ranks)
    top1 = float(np.mean(ranks_a == 1))
    mrr = float(np.mean(np.where(ranks_a > 0, 1.0 / np.maximum(ranks_a, 1), 0.0)))
    cell = CellResult(
        axis=axis, label=label, n_queries=len(records),
        top1_accuracy=top1, mrr=mrr,
        median_score=float(np.median(top_scores)),
        median_true_score=float(np.median(true_scores)),
    )
    return cell, records


def _threshold_analysis(
    genuine: list[tuple[int, float, float]],
    impostor_scores: list[float],
) -> tuple[list[dict], dict]:
    """Sweep the accept threshold; report TAR and FAR at each.

    ``genuine`` records are ``(rank, top_score, true_score)`` for in-corpus
    queries pooled across a range of degradations. An accept is **correct** only
    if the top candidate is the right track *and* its score clears the
    threshold; a genuine query whose top-1 is the wrong track counts against
    true-accept even if it clears threshold (that is a substitution error, and
    pretending otherwise would flatter the system).

    ``impostor_scores`` are top-1 scores for out-of-corpus queries. Any impostor
    score at or above the threshold is a **false accept** — by construction the
    correct answer is "not in database", so *any* confident answer is wrong.

    Returns the full table plus a chosen operating point: the lowest threshold
    whose false-accept rate is <= 1%, which is the usual way to pick an operating
    point when a false accept is much more costly than a miss (naming the wrong
    song is worse than saying "I don't know").
    """
    gen_correct_scores = np.asarray(
        [top for rank, top, _ in genuine if rank == 1], dtype=np.float64
    )
    n_gen = len(genuine)
    imp = np.asarray(impostor_scores, dtype=np.float64)

    candidates = sorted(set(range(0, 61, 2)))
    table: list[dict] = []
    for thr in candidates:
        # true-accept as a fraction of *all* genuine queries: a query counts as
        # accepted only if its top-1 is correct AND clears the threshold.
        tar_all = float(np.sum(gen_correct_scores >= thr)) / max(n_gen, 1)
        far = float(np.mean(imp >= thr)) if imp.size else 0.0
        table.append(
            dict(threshold=thr, true_accept_rate=round(tar_all, 4),
                 false_accept_rate=round(far, 4))
        )

    op = {}
    for entry in table:
        if entry["false_accept_rate"] <= 0.01:
            op = dict(entry)
            op["policy"] = "lowest threshold with FAR <= 1%"
            break
    if not op and table:
        op = dict(table[-1])
        op["policy"] = "FAR never reached 1%; reporting strictest threshold"
    return table, op


def build_index(corpus: synth.Corpus) -> tuple[FingerprintIndex, float]:
    """Fingerprint and index every track; return the index and build time."""
    index = FingerprintIndex()
    t0 = time.perf_counter()
    for spec in corpus.specs:
        index.add_track(spec.track_id, spec.name, _fp(corpus.audio[spec.track_id]))
    index.freeze()
    return index, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> EvalReport:
    """Run the full evaluation and print a report. Returns the report object."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tracks", type=int, default=48)
    parser.add_argument("--n-impostors", type=int, default=24)
    parser.add_argument("--track-seconds", type=float, default=25.0)
    parser.add_argument("--queries-per-track", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument(
        "--quick", action="store_true",
        help="Small fast configuration for CI (fewer tracks and cells).",
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.n_tracks = min(args.n_tracks, 12)
        args.n_impostors = min(args.n_impostors, 6)
        args.track_seconds = min(args.track_seconds, 15.0)

    t_start = time.perf_counter()
    print("=" * 74)
    print("audio-fingerprint — retrieval evaluation on a SYNTHETIC corpus")
    print("(numbers are an UPPER BOUND on real audio; see synth.py / README)")
    print("=" * 74)
    print(
        f"building {args.n_tracks} reference tracks x {args.track_seconds:.0f}s "
        f"+ {args.n_impostors} impostor tracks ..."
    )

    corpus = synth.make_corpus(
        args.n_tracks, duration_s=args.track_seconds, seed=args.seed
    )
    impostors = synth.make_impostor_corpus(
        args.n_impostors, duration_s=args.track_seconds, seed=args.seed + 555
    )
    index, build_s = build_index(corpus)
    stats = index.stats()
    print(
        f"indexed in {build_s:.1f}s: {stats.n_distinct_hashes:,} keys, "
        f"{stats.n_postings:,} postings, mean posting {stats.mean_posting_length:.2f}"
    )

    report = EvalReport(
        config=dict(
            n_tracks=args.n_tracks, n_impostors=args.n_impostors,
            track_seconds=args.track_seconds, queries_per_track=args.queries_per_track,
            seed=args.seed, quick=args.quick,
            sample_rate=CFG.sample_rate, n_fft=CFG.n_fft, hop_length=CFG.hop_length,
            target_density=DEFAULT_PEAK_CONFIG.target_density,
            fan_out=DEFAULT_HASH_CONFIG.fan_out,
            freq_quant=DEFAULT_HASH_CONFIG.freq_quant,
        ),
        index_stats=asdict(stats),
    )

    rng = np.random.default_rng(args.seed + 1)
    pooled_genuine: list[tuple[int, float, float]] = []

    # --- Sweep 1: SNR, at a fixed 5 s excerpt, full band -------------------
    snr_grid = [40.0, 20.0, 10.0, 5.0, 0.0, -5.0] if not args.quick else [20.0, 5.0, -5.0]
    print("\nSNR sweep (5 s excerpt, full band):")
    for snr in snr_grid:
        cell, recs = _run_cell(
            index, corpus, rng, "snr", f"{snr:+.0f} dB",
            excerpt_s=5.0, snr_db=snr, band=None,
            queries_per_track=args.queries_per_track,
        )
        report.snr_sweep.append(cell)
        pooled_genuine.extend(recs)
        print("  " + cell.row())

    # --- Sweep 2: excerpt length, at a fixed 10 dB SNR ---------------------
    exc_grid = [1.0, 2.0, 3.0, 5.0, 10.0] if not args.quick else [1.0, 3.0, 10.0]
    print("\nExcerpt-length sweep (10 dB SNR, full band):")
    for exc in exc_grid:
        cell, recs = _run_cell(
            index, corpus, rng, "excerpt", f"{exc:.0f} s",
            excerpt_s=exc, snr_db=10.0, band=None,
            queries_per_track=args.queries_per_track,
        )
        report.excerpt_sweep.append(cell)
        pooled_genuine.extend(recs)
        print("  " + cell.row())

    # --- Sweep 3: band-limit, at a fixed 5 s excerpt, 15 dB SNR ------------
    bands: list[tuple[str, tuple[float | None, float | None] | None]] = [
        ("full band", None),
        ("telephone 300-3400", (300.0, 3400.0)),
        ("highpass 500", (500.0, None)),
        ("lowpass 2000", (None, 2000.0)),
    ]
    if args.quick:
        bands = bands[:2]
    print("\nBand-limit sweep (5 s excerpt, 15 dB SNR):")
    for label, band in bands:
        cell, recs = _run_cell(
            index, corpus, rng, "band", label,
            excerpt_s=5.0, snr_db=15.0, band=band,
            queries_per_track=args.queries_per_track,
        )
        report.bandlimit_sweep.append(cell)
        pooled_genuine.extend(recs)
        print("  " + cell.row())

    # --- Impostors: same degradation distribution, out-of-corpus tracks ----
    print("\nImpostor queries (out-of-corpus; mixed degradation):")
    imp_scores: list[float] = []
    imp_conditions = (
        [(5.0, 10.0), (3.0, 5.0), (5.0, 0.0)] if not args.quick else [(5.0, 5.0)]
    )
    for exc, snr in imp_conditions:
        for spec in impostors.specs:
            q = _degrade(
                impostors.audio[spec.track_id], rng,
                excerpt_s=exc, snr_db=snr, band=None,
            )
            fp = _fp(q)
            results = identify(index, fp, DEFAULT_MATCH_CONFIG, CFG)
            imp_scores.append(results[0].score if results else 0.0)
    imp_arr = np.asarray(imp_scores, dtype=np.float64)
    report.impostor_summary = dict(
        n_impostor_queries=int(imp_arr.size),
        max_score=float(imp_arr.max()) if imp_arr.size else 0.0,
        p95_score=float(np.percentile(imp_arr, 95)) if imp_arr.size else 0.0,
        median_score=float(np.median(imp_arr)) if imp_arr.size else 0.0,
    )
    print(
        f"  {imp_arr.size} impostor queries: "
        f"median score {report.impostor_summary['median_score']:.0f}, "
        f"p95 {report.impostor_summary['p95_score']:.0f}, "
        f"max {report.impostor_summary['max_score']:.0f}"
    )

    # --- Threshold / false-accept analysis ---------------------------------
    table, op = _threshold_analysis(pooled_genuine, imp_scores)
    report.threshold_table = table
    report.operating_point = op
    print("\nThreshold analysis (genuine queries pooled across all sweeps):")
    print("  thr | true-accept | false-accept")
    for e in table:
        if e["threshold"] % 4 == 0:
            print(
                f"  {e['threshold']:>3} |   {e['true_accept_rate']:>6.3f}    |   "
                f"{e['false_accept_rate']:>6.3f}"
            )
    if op:
        print(
            f"\nchosen operating point: threshold={op['threshold']} "
            f"-> true-accept {op['true_accept_rate']:.3f}, "
            f"false-accept {op['false_accept_rate']:.3f}  ({op['policy']})"
        )

    report.runtime_seconds = time.perf_counter() - t_start
    print(f"\ntotal runtime: {report.runtime_seconds:.1f}s")

    if args.json_out:
        out = dict(
            config=report.config, index_stats=report.index_stats,
            snr_sweep=[asdict(c) for c in report.snr_sweep],
            excerpt_sweep=[asdict(c) for c in report.excerpt_sweep],
            bandlimit_sweep=[asdict(c) for c in report.bandlimit_sweep],
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
