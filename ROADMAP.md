# Roadmap

Status: what is built and measured, and what is not. The emphasis is on the gap
between the synthetic and real evaluations, since that gap decides whether this is
a portfolio toy or the start of something usable.

## Where the evaluation stands

The headline evaluation is on a real Creative-Commons music corpus (36 indexed +
12 impostor tracks; see the README's "Real audio" section and
`results/eval_real.json`). Its result: for same-recording identification, real
music is *not* harder than synthetic tones, written up in the decision doc. The
synthetic corpus of harmonic tones runs the same pipeline as a controlled stress
test and characterises the algorithm rather than real-world accuracy. The
remaining gap is no longer "synthetic vs real"; it is same-recording vs
cross-recording (covers/live/re-records) and scale, neither of which either
evaluation tests. Read nothing here as a claim about cross-recording matching or
million-track corpora.

---

## Done, and measured

- [x] **STFT front-end from scratch.** Own framing, Hann windowing, dB scaling;
      linear-frequency (not mel) with the reasoning documented. `numpy`/`scipy`
      only, no `librosa`.
- [x] **Density-controlled peak picking.** Local maxima + adaptive local
      threshold + per-(second × frequency-band) quota. Gain-invariant to the bit;
      density held near target across a 80 dB gain range. *Verified.*
- [x] **Constellation hashing.** Anchor + target-zone pairing, `(f1, f2, Δt)`
      packed to int64, deterministic. Fan-out and quantisation chosen by
      measurement. *Verified deterministic.*
- [x] **Inverted index.** CSR-style flat arrays, vectorised whole-query lookup,
      `.npz` round-trip without pickle. *Round-trip verified element-by-element.*
- [x] **Offset-histogram voting.** Score = tallest aligned bin, recovers
      alignment, beats a raw-count baseline (0.958 vs 0.729 measured; a
      constructed near-miss test pins the direction).
- [x] **Evaluation harness.** SNR / excerpt-length / band-limit sweeps, top-1
      accuracy + MRR, plus an impostor set and a threshold/false-accept table.
      Runs in CI (`--quick`).
- [x] **Tests + CI.** 40 tests; lint + tests + evaluation on 3.10/3.11/3.12.

---

## Next, the part that actually matters

Ordered by how much each would change the honesty of the results.

### 1. Real audio: done (and it did not go as predicted)
Built `src/audio_fingerprint/corpus.py` (Internet Archive `netlabels`,
CC-licensed, attributed in `DATA_SOURCES.md`) and `tests/evaluate_real.py`, and
re-ran the identical harness on 36 real indexed tracks + 12 impostors, plus a new
MP3 codec round-trip axis. The numbers did not fall: top-1 stays at/near
1.000 across noise to 0 dB, band-limiting, and MP3 to 32 kbps, and *beats*
synthetic on short excerpts. Reason: this is *same-recording* identification, so
real music's peak jitter is identical on both sides and its spectral richness
helps. The open problem is therefore **cross-recording** matching, not real-vs-
synthetic. Next real steps below (#2–#6) are re-scoped around that.

### 2. Re-tune every hyperparameter on real audio
Several defaults were chosen by measurement on synthetic audio and are known to
be data-dependent:
- **`freq_quant = 1` (no frequency quantisation)** almost certainly flips to 2 on
  real audio. Synthetic partials are perfectly stationary so there is no bin
  jitter to absorb; vibrato, inharmonicity, a turntable running 0.3% fast, and
  MP3's own quantisation all introduce jitter that quantisation is designed to
  tolerate. This is the clearest example of a knob a synthetic eval cannot set.
- **Peak density (30/s)** and **fan-out (8)** may need to rise, because real
  spectral peaks are less repeatable, so more redundancy is needed to hold recall.
- The **accept threshold** will need re-derivation entirely, because real impostor
  score distributions are wider (real tracks share drum samples, chord loops,
  mastering chains), so the threshold that gives 0% false accepts here will not
  there.

### 3. Realistic degradations
The current model covers additive white noise, Butterworth band-limiting,
cropping, gain, resample-based speed change, and OLA time-stretch. Missing, in
rough order of importance for a phone/vinyl use case:
- **MP3/AAC codec round-trip.** Codecs quantise in exactly the time-frequency
  domain the fingerprint lives in; this is the single most important missing axis.
- **Convolutional reverb** (room impulse responses), which smears peaks in time.
- **Dynamic-range compression.** The one degradation that partly defeats the
  gain-invariance argument, because it is a *non-linear*, level-dependent gain.
- **Coloured / real-world noise** (café, traffic, crowd) instead of white. This
  is actually *easier* for a banded picker than white noise, so adding it would
  make some results look better; worth having for honesty in both directions.
- **Vinyl-specific**: surface clicks, wow/flutter, RIAA response, groove wear.

### 4. Phase-vocoder time-stretch
The current `time_stretch` is plain overlap-add and introduces some phasiness. A
phase vocoder would be a cleaner model of a broadcast time-compressor. Low
priority, since it preserves partial frequencies, which is what the fingerprint
needs.

### 5. Scale
Sixty tracks is not sixty million. False-accept pressure grows with corpus size.
Worth testing: does the current hash entropy hold up at 10^4–10^5 tracks, and at
what point does the posting-length stop-word cut (`max_posting_length`, built but
off by default) become necessary? This needs #1 first to be meaningful.

### 6. Needle-drop matching demo
The motivating vinyl use case: rip a record, match it against a reference
pressing, use the recovered alignment offset to locate and stitch. The pipeline
already returns the offset; this is an application on top, blocked on #1.

---

## Explicitly out of scope

- Beating or matching a commercial system. Different problem (scale, real audio,
  latency, adversarial conditions) and not the point of a portfolio project.
- A learned/embedding-based fingerprinter (e.g. neural audio embeddings). A
  worthwhile *contrast* project, but this one is deliberately the classical,
  fully-interpretable landmark approach, where every decision is inspectable,
  which is the pedagogical and interview value.
- Real-time / streaming identification. The index supports it; the harness does
  not exercise it.
