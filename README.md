# audio-fingerprint

**Identify a recording from a short, noisy excerpt** — the problem Shazam solves.
Given a few seconds of degraded audio (background noise, a limited-bandwidth
microphone, a random start point, a different volume), find which track in a
reference database it came from, and *where* in that track it starts.

This repository implements the classic landmark-fingerprinting pipeline from
scratch on `numpy`/`scipy` — spectrogram peak-picking, combinatorial
constellation hashing, an inverted index, and time-offset histogram voting —
and, more importantly, **measures how well it actually works** under controlled
degradation, including a false-accept analysis against out-of-database
impostors.

> ### ⚠️ The evaluation corpus is SYNTHETIC
> There is no music here. The "tracks" are procedurally generated harmonic tone
> sequences (see [`src/audio_fingerprint/synth.py`](src/audio_fingerprint/synth.py)).
> Retrieval on synthetic audio is **substantially easier** than on real music —
> synthetic partials are stationary, tracks are spectrally distinct by
> construction, there is no codec/reverb/microphone chain, and the corpus is 48
> tracks, not 48 million. **Every number below is an upper bound on real-world
> behaviour and must not be read as comparable to a commercial system.** What the
> numbers *do* show is how the algorithm degrades along each axis, which is the
> engineering question this project is built to answer. See
> [ROADMAP.md](ROADMAP.md) for what real audio would change.

---

## The real result table

All numbers are produced by
[`tests/evaluate.py`](tests/evaluate.py) on the actual pipeline — nothing here is
hand-written. Reproduce with:

```bash
python tests/evaluate.py --n-tracks 48 --track-seconds 25 --queries-per-track 2 \
    --json-out results/eval_full.json
```

**Database:** 48 synthetic tracks × 25 s → 120,986 distinct hash keys, 255,121
postings (mean posting length 2.11). Analysis: 11,025 Hz, 1024-pt FFT, 256 hop,
30 peaks/s, fan-out 8. Full run: ~289 s on one laptop core. Seed `2024`.

### Robustness to additive noise (5 s excerpt, full band)

| SNR    | queries | top-1 acc | MRR   | median score |
|--------|--------:|----------:|------:|-------------:|
| +40 dB |      96 | **1.000** | 1.000 |          152 |
| +20 dB |      96 | **1.000** | 1.000 |          109 |
| +10 dB |      96 | **1.000** | 1.000 |           69 |
| +5 dB  |      96 | **1.000** | 1.000 |           59 |
|  0 dB  |      96 | **1.000** | 1.000 |           40 |
| −5 dB  |      96 | **0.969** | 0.972 |           18 |

At 0 dB the noise is as loud as the signal, and identification is still perfect
on this corpus. Only when noise exceeds the signal (−5 dB) does it start to slip.

### Sensitivity to excerpt length (10 dB SNR, full band)

| excerpt | queries | top-1 acc | MRR   | median score |
|---------|--------:|----------:|------:|-------------:|
|   1 s   |      96 | **0.823** | 0.893 |           10 |
|   2 s   |      96 | **0.969** | 0.977 |           29 |
|   3 s   |      96 | **1.000** | 1.000 |           46 |
|   5 s   |      96 | **1.000** | 1.000 |           73 |
|  10 s   |      96 | **1.000** | 1.000 |          148 |

This is the sharpest axis: a 1-second excerpt yields only ~10 aligned votes and
recall drops to 82%; by 3 seconds there is enough evidence for perfect top-1.

### Robustness to band-limiting (5 s excerpt, 15 dB SNR)

| filter               | queries | top-1 acc | MRR   | median score |
|----------------------|--------:|----------:|------:|-------------:|
| full band            |      96 | **1.000** | 1.000 |           87 |
| telephone 300–3400   |      96 | **1.000** | 1.000 |           73 |
| high-pass 500 Hz     |      96 | **1.000** | 1.000 |           56 |
| low-pass 2000 Hz     |      96 | **1.000** | 1.000 |           82 |

Band-limiting is where landmark fingerprinting is designed to shine: peaks are
spread across frequency bands by the picker, so removing a band costs only the
hashes in that band rather than breaking the fingerprint. Every band-limit here
is survived perfectly. (This is also where the synthetic corpus flatters the
system most — a real phone microphone response is far less clean than a
Butterworth filter.)

### False accepts — the part that separates a retrieval system from a toy

72 **impostor** queries drawn from 24 tracks that were *never indexed*. By
construction the correct answer is "not in the database", so any confident answer
is a false accept. Genuine queries are pooled across all sweeps above (1,440
queries). Sweeping the accept threshold on the match score:

| score threshold | true-accept rate | false-accept rate |
|-----------------|-----------------:|------------------:|
|  0              |            0.984 |             1.000 |
|  8              |            0.967 |             0.653 |
| 12              |            0.928 |             0.111 |
| 16              |            0.886 |             0.028 |
| **20**          |        **0.842** |         **0.000** |
| 24              |            0.802 |             0.000 |
| 32              |            0.721 |             0.000 |

Impostor scores: median 8, p95 13, **max 18**. Genuine matches clear 40–150 on
easy conditions but fall to ~10 on the hardest (1 s excerpts). **The chosen
operating point is threshold = 20** — the lowest threshold that drives the
false-accept rate to zero on this corpus — giving an 84.2% true-accept rate. The
trade is explicit: pushing the threshold down to 16 recovers ~4 points of recall
but readmits a 2.8% false-accept rate. Naming the wrong song is worse than saying
"I don't know", so the operating point favours precision. See
[the decision doc](../decisions/audio-fingerprint-decisions.md) for the full
argument.

### Why offset voting, not hash overlap

The matcher scores on the height of the tallest time-offset histogram bin, not on
how many hashes a query shares with a track. The difference is measurable
([`examples/demo.py`](examples/demo.py), 48 tracks, 2 s excerpts at 0 dB SNR):

| ranker                       | top-1 accuracy |
|------------------------------|---------------:|
| **offset-histogram voting**  |      **0.958** |
| raw hash-count baseline      |          0.729 |

Both rankers see identical hashes. Voting wins because two unrelated tracks share
plenty of local spectral geometry by chance — but only the *correct* track has
those collisions land at a single consistent time offset. Alignment structure,
not overlap volume, is the discriminative signal.
[`tests/test_matching.py`](tests/test_matching.py) locks this in with a
constructed near-miss where the raw-count baseline provably picks the wrong track
and voting recovers the right one.

---

## How it works

```
audio ─▶ STFT (dB, linear freq)   spectrogram.py   own framing/windowing, no librosa
      ─▶ spectral peaks           peaks.py         local maxima + adaptive threshold + density quota
      ─▶ paired hashes            hashing.py       (f1, f2, Δt) constellation pairs, fan-out 8
      ─▶ inverted index           index.py         hash ─▶ [(track_id, offset)], CSR layout
      ─▶ offset-histogram vote    match.py         score = tallest aligned bin; also recovers position
```

The design decisions — peak density, fan-out, hash quantisation, why offset
voting, and how the threshold is chosen — are written up in
[**`../decisions/audio-fingerprint-decisions.md`**](../decisions/audio-fingerprint-decisions.md),
with the measurements behind each one. Read that before the code.

Key robustness properties, each verified by a test:

- **Gain invariance is exact.** Working in dB turns a volume change into an
  additive constant, which moves neither local maxima nor a local-statistics
  threshold. Peaks are *bit-identical* from −40 dB to +40 dB
  (`test_peaks.py::test_peaks_are_gain_invariant`).
- **Peak density is pinned near target regardless of loudness**, so the match
  score means the same thing across excerpts
  (`test_peaks.py::test_density_near_target_across_signal_levels`).
- **Hashes are translation-invariant**, so a query cut from anywhere in a track
  produces the same keys (`test_hashing.py::test_hashes_are_translation_invariant`).

---

## Quick start

```bash
pip install -e ".[dev]"

python examples/demo.py               # identify a noisy excerpt; voting vs raw count
pytest -q                             # 40 tests: determinism, round-trip, voting, density
python tests/evaluate.py --quick      # fast end-to-end evaluation (CI configuration)
python tests/evaluate.py              # the full run behind the table above (~5 min)
```

Minimal library use:

```python
import numpy as np
from audio_fingerprint import fingerprint_audio, FingerprintIndex, identify, synth

corpus = synth.make_corpus(20, duration_s=20.0, seed=0)   # SYNTHETIC audio
index = FingerprintIndex()
for spec in corpus.specs:
    index.add_track(spec.track_id, spec.name,
                    fingerprint_audio(corpus.audio[spec.track_id]))

rng = np.random.default_rng(0)
excerpt, start_s = synth.crop(corpus.audio[3], corpus.sample_rate, 4.0, rng)
noisy = synth.add_noise(excerpt, snr_db=5.0, rng=rng)

best = identify(index, fingerprint_audio(noisy))[0]
print(best.name, f"@ {best.offset_seconds:.2f}s  score={best.score:.0f}")
```

---

## Connection to vinyl

The motivating real use case (not yet built — see the roadmap) is matching a
**needle-drop rip** of a record against a reference pressing. That is why the
degradation model includes `speed_change` (a turntable running slightly fast
shifts pitch *and* tempo together) and why the matcher returns an alignment
offset, not just a track id: locating *where* a rip sits in a reference is what
lets you stitch, de-click against a clean copy, or verify a pressing.

---

## Status

Work in progress. The pipeline, the evaluation harness, the tests and CI are real
and running; the results above are reproducible on any clone. What is **not** here
is real audio — the entire corpus is synthetic, and the honest limitations of
that are stated above, in [ROADMAP.md](ROADMAP.md), and in the module docstrings.
See [LICENSE](LICENSE) (MIT).
