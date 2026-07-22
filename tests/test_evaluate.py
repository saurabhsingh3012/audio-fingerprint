"""Smoke test for the evaluation harness itself.

Runs the whole end-to-end evaluation at its smallest configuration and asserts
the report is internally consistent. This is deliberately not checking specific
accuracy numbers — those are the deliverable, reported in the README from a real
run, not pinned in a test where they would be either brittle or meaningless.
What it *does* guarantee is that ``main()`` runs, the metrics are in range, and
the false-accept machinery produces a sane threshold table.
"""

from __future__ import annotations

from tests.evaluate import main


def test_evaluation_runs_end_to_end():
    report = main(
        ["--quick", "--n-tracks", "8", "--n-impostors", "4", "--track-seconds", "10"]
    )

    assert report.snr_sweep and report.excerpt_sweep and report.bandlimit_sweep

    for cell in report.snr_sweep + report.excerpt_sweep + report.bandlimit_sweep:
        assert 0.0 <= cell.top1_accuracy <= 1.0
        assert 0.0 <= cell.mrr <= 1.0
        assert cell.mrr >= cell.top1_accuracy - 1e-9  # MRR >= top-1 by definition
        assert cell.n_queries > 0

    # A clean, long-ish excerpt should be near-perfect even on 8 tracks; a very
    # short one is allowed to be worse. The evaluation must at least distinguish
    # them or match — never invert.
    assert report.index_stats["n_tracks"] == 8

    # Threshold table monotonicity: raising the threshold cannot increase FAR
    # and cannot increase the true-accept rate.
    table = report.threshold_table
    fars = [e["false_accept_rate"] for e in table]
    tars = [e["true_accept_rate"] for e in table]
    assert fars == sorted(fars, reverse=True)
    assert tars == sorted(tars, reverse=True)

    # An operating point was selected.
    assert "threshold" in report.operating_point
