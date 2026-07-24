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

> ### ⚠️ The default evaluation corpus is SYNTHETIC — but there is now a real one too
> The tables in the next section use procedurally generated harmonic tone
> sequences, not music (see [`src/audio_fingerprint/synth.py`](src/audio_fingerprint/synth.py)),
> and are an **upper-bound characterisation of the algorithm**, not a claim about
> real-world accuracy. A companion evaluation on **real Creative-Commons music**
> (36 indexed tracks + 12 impostors from the Internet Archive) now runs the
> identical pipeline and is reported in **[§ Real audio](#real-audio)** below,
> with an honest comparison. The synthetic numbers are kept, clearly labelled;
> the real ones are added alongside. See [ROADMAP.md](ROADMAP.md) for what is
> still untested.

---

## Measured results — SYNTHETIC corpus

All numbers in this section are produced by
[`tests/evaluate.py`](tests/evaluate.py) on the actual pipeline — nothing here is
hand-written. **The corpus is synthetic** (see the warning above); the real-audio
counterpart is [§ Real audio](#real-audio). Reproduce with:

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

## Real audio

The tables above characterise the algorithm on synthetic tones. This section runs
the **identical pipeline and the identical degradation axes** on **real
Creative-Commons music**: 36 indexed tracks (90.6 min) plus 12 held-out impostor
tracks, downloaded from the Internet Archive [`netlabels`](https://archive.org/details/netlabels)
collection. Every track's title, creator, exact CC license and source item are
listed in [DATA_SOURCES.md](DATA_SOURCES.md). No audio is committed — it is cached
under `data/` (git-ignored); only the code, the attribution table, and the
measured [`results/eval_real.json`](results/eval_real.json) are in the repo.

Reproduce (needs network access and an `ffmpeg` binary):

```bash
pip install -e ".[real]"
python -m audio_fingerprint.corpus                      # ~48 CC tracks -> data/ (git-ignored)
python tests/evaluate_real.py --json-out results/eval_real.json
```

**Database:** 36 real tracks → **445,398** distinct hash keys, **1,153,980**
postings (mean posting 2.59) — vs 120,986 keys for 48 synthetic tracks, because
real music is spectrally much denser. 72 queries per cell (2 per track). Seed
`2024`. Queries are degraded excerpts of indexed tracks; impostor queries are
excerpts of the 12 tracks that were **never** indexed.

### The headline, stated honestly up front

The whole README predicted real music would be *harder* than synthetic tones.
**On this task, it is not.** Top-1 accuracy is at or near **1.000** across
additive noise down to 0 dB, every band-limit, and an MP3 round-trip down to
32 kbps — and it is *better* than synthetic on the hardest short-excerpt cell
(1 s: **0.986 real vs 0.823 synthetic**). Two structural reasons, both important
for reading the number correctly:

- **It is same-recording identification.** The query is a degraded excerpt of the
  *same* recording that sits in the index. The peak jitter the synthetic caveats
  worry about — vibrato, inharmonicity, formant motion — is *identical* in the
  reference and the query because it is literally the same performance, so it
  never desynchronises the two fingerprints. This is exactly the Shazam use case
  (a noisy capture of a master that is in the database), and it is the case
  landmark fingerprinting is strongest at.
- **Real music is spectrally richer**, so it emits *more* landmarks per second
  than a sparse tone sequence. Median match scores run 2–3× the synthetic ones,
  and a short noisy excerpt therefore has more aligned votes to work with.

So the synthetic corpus was **not** the loose upper bound it was billed as; for
same-recording retrieval it is a fair proxy. The regime where real audio would
genuinely break this system is a *different* one that **neither** evaluation
tests — see [§ What this still does not measure](#what-this-still-does-not-measure).

### Real vs synthetic, side by side (top-1 accuracy)

Additive noise, 5 s excerpt, full band:

| SNR    | synthetic | real  |
|--------|----------:|------:|
| +20 dB |     1.000 | 1.000 |
| +10 dB |     1.000 | 1.000 |
| +5 dB  |     1.000 | 0.986 |
|  0 dB  |     1.000 | 1.000 |
| −5 dB  |     0.969 | 0.972 |

Excerpt length, 10 dB SNR, full band:

| excerpt | synthetic | real  |
|---------|----------:|------:|
|   1 s   |     0.823 | 0.986 |
|   2 s   |     0.969 | 1.000 |
|   3 s   |     1.000 | 0.972 |
|   5 s   |     1.000 | 1.000 |
|  10 s   |     1.000 | 1.000 |

Real short excerpts *beat* synthetic (more spectral content per second). The 3 s
real dip (0.972 = 2 misses in 72) below the 5 s cell is sampling noise: each cell
draws independent random crops, and a couple of near-silent intro/outro excerpts
land in the 3 s cell. Band-limiting (telephone 300–3400, high-pass 500, low-pass
2000) is survived at **1.000** top-1 in every case, as on synthetic.

### MP3 codec round-trip — the axis the synthetic harness structurally could not test

Each query is re-encoded to MP3 via `ffmpeg` at a controlled bitrate and decoded
back (5 s excerpt, 10 dB SNR, full band). The design doc called this "the single
most important missing robustness axis." Measured:

| codec         | top-1 | median score |
|---------------|------:|-------------:|
| no codec      | 1.000 |          229 |
| MP3 128 kbps  | 1.000 |          257 |
| MP3 64 kbps   | 1.000 |          262 |
| MP3 48 kbps   | 1.000 |          214 |
| MP3 32 kbps   | 1.000 |          202 |

Top-1 is unmoved down to 32 kbps; codec quantisation barely touches the strongest
landmarks (and, since the reference is itself decoded from the Archive.org MP3,
both sides share codec provenance). This *validates* the "peaks are the robust
thing" premise — while being a much gentler result than the doc feared. It is a
**second** MP3 pass, not a first encode of lossless masters, and it does not test
AAC/Opus or an actual phone microphone; those remain open.

### False accepts

36 impostor queries drawn from the 12 never-indexed tracks (mixed degradation
including MP3 64 kbps). Genuine queries pooled across all sweeps (1,512 queries).

| statistic                 | synthetic | real |
|---------------------------|----------:|-----:|
| impostor median score     |         8 |    6 |
| impostor p95 score        |        13 |   10 |
| impostor **max** score    |        18 |   10 |
| operating-point threshold |        20 |  12  |
| true-accept at that point |     0.842 | **0.982** |
| false-accept at that point|     0.000 | 0.000 |

Real impostors are **better** separated than synthetic ones (max score 10 vs 18),
so the false-accept rate reaches zero at a *lower* threshold (12 vs 20) while the
true-accept rate is *higher* (98.2% vs 84.2%). This is the opposite of the doc's
prediction that real impostor distributions would be wider — because this corpus
is deliberately diverse (one track per netlabel release, no covers or shared
samples). That is also the caveat: **12 impostors is a small, easy set** that does
not probe the cover/sample/shared-master confusion where real false-accepts
actually come from.

### freq_quant 1 vs 2 — testing a prediction the design doc made

The design doc predicted that frequency quantisation (`freq_quant = 2`), which did
*not* help on stationary synthetic partials, would "almost certainly flip" to help
on real audio. At a deliberately hard cell (3 s excerpt, 0 dB SNR, MP3 64 kbps):

| freq_quant | top-1 | distinct keys |
|-----------:|------:|--------------:|
| 1          | 1.000 |       445,398 |
| 2          | 1.000 |       248,639 |

Both are perfect; `freq_quant = 2` simply uses **44% fewer keys**. So on this
*same-recording* task quantisation neither helps nor hurts accuracy — the bin
jitter it exists to absorb comes from *cross-recording* differences that are not
present here. The prediction's *reasoning* was about a regime this eval does not
exercise; measured in-regime, quantisation is a free index shrink, not an accuracy
lever. (Getting the query and index to use the *same* `freq_quant` matters: an
early version fingerprinted the query at `freq_quant=1` against a `freq_quant=2`
index and top-1 collapsed to 0.11 — a config-mismatch bug, now fixed and worth
flagging because it is an easy mistake to make silently.)

### What this still does not measure

The honest limitations, because the near-perfect numbers above are for one
specific, favourable regime:

- **Cross-recording matching.** Every genuine query here is the *same recording*
  as its reference. Covers, live versions, re-records and alternate masters —
  where the peaks genuinely desynchronise — are the case that would actually hurt,
  and are untested. This, not "synthetic vs real", is where the real difficulty
  lives.
- **Scale.** 36 tracks, not 36 million. False-accept pressure grows with corpus
  size; the clean impostor separation here will not survive it.
- **A harder impostor set.** 12 unrelated tracks is too easy and too small; the
  real test is impostors that *share* drum samples, chord loops and mastering.
- **First-encode and other codecs / a real microphone.** The codec test is a
  second MP3 pass on already-MP3 audio, not AAC/Opus, not acoustic capture.

The one thing that clearly carried over from synthetic to real is the **core
mechanism**: offset-histogram voting separates genuine from impostor cleanly on
both, because that is a geometric fact about aligned collisions, independent of
what the audio is.

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
python tests/evaluate.py              # the full synthetic run (~5 min)

# Real Creative-Commons audio (needs network + ffmpeg; not run in CI):
pip install -e ".[real]"
python -m audio_fingerprint.corpus    # download ~48 CC tracks -> data/ (git-ignored)
python tests/evaluate_real.py         # the run behind the "Real audio" section
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

Work in progress. The pipeline, both evaluation harnesses, the tests and CI are
real and running; every result is reproducible on any clone. The project now runs
on **both** a synthetic corpus (the default, in CI) **and** a real Creative-Commons
music corpus ([§ Real audio](#real-audio), fetched on demand, kept out of CI
because it needs network). The honest remaining limitations — cross-recording
matching, scale, harder impostors, non-MP3 codecs and real microphones — are
stated in [§ What this still does not measure](#what-this-still-does-not-measure),
[ROADMAP.md](ROADMAP.md), and the module docstrings. See [LICENSE](LICENSE) (MIT).
