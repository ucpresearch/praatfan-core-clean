# CLAUDE.md - praatfan-core-clean

## Project Overview

**praatfan-core-clean** - A clean room reimplementation of Praat's core acoustic analysis algorithms.

---

## ⚠️ CRITICAL: Clean Room Constraints

This is a **clean room implementation**. The entire point is to create a non-GPL implementation by working ONLY from public documentation.

### You must NOT:
- Read, reference, or examine any GPL-licensed Praat source code
- Copy algorithms from the existing `praatfan-core-rs` implementation
- Use any information that could only be obtained from Praat's source code
- Include specific numeric constants that aren't documented (find them via black-box testing instead)

### You MAY use:
- Published academic papers (see `docs/RESOURCES.md`)
- Praat's official user-facing documentation (manual pages)
- Standard signal processing textbooks
- **Black-box testing against Praat/parselmouth output** - this is the primary validation method

### Why this matters:
The goal is a **non-GPL license** (MIT or Apache-2.0). This requires that the implementation be based solely on:
- Published academic papers describing the algorithms
- Praat's public documentation (which describes behavior, not implementation)
- Standard signal processing techniques from textbooks

If we use GPL source code, the result must also be GPL.

---

## 🎯 Mono Audio Only

This implementation supports **mono audio only**.

- Multi-channel files should error or require explicit channel selection
- This eliminates undocumented multi-channel handling concerns
- Simplifies the implementation significantly

---

## 🎵 Supported Audio Formats

praatfan uses **soundfile/libsndfile** for audio I/O, supporting a wide range of formats:

| Format | Extensions | Notes |
|--------|------------|-------|
| WAV | `.wav` | PCM 8/16/24/32-bit, 32/64-bit float, u-law, a-law |
| FLAC | `.flac` | Free Lossless Audio Codec |
| MP3 | `.mp3` | Requires libsndfile 1.1.0+ |
| OGG Vorbis | `.ogg` | Lossy compressed |
| AIFF | `.aiff`, `.aif` | Apple audio format |
| AU | `.au` | Sun/NeXT audio |
| CAF | `.caf` | Core Audio Format |

**Note:** Multi-channel files require explicit channel selection via `Sound.from_file_channel(path, channel=0)`.

---

## 🔄 Parselmouth Compatibility

The praatfan package provides full parselmouth compatibility with backend switching.

### Drop-in Replacement for parselmouth

Use for **minimal code changes** when migrating from parselmouth. The `Sound()` constructor matches parselmouth's API:

```python
# Before (parselmouth)
import parselmouth
from parselmouth.praat import call
snd = parselmouth.Sound("audio.wav")
pitch = call(snd, "To Pitch (ac)", 0, 75, 600)

# After (praatfan) — just change imports
from praatfan import Sound, call
snd = Sound("audio.wav")  # Same constructor syntax
pitch = call(snd, "To Pitch (ac)", 0, 75, 600)
```

Also supports backend switching:
```python
from praatfan import set_backend
set_backend("parselmouth")  # or "praatfan", "praatfan_rust", "praatfan_gpl"
```

### Clean API

The same package also supports a cleaner, more Pythonic API:

```python
from praatfan import Sound

snd = Sound("audio.wav")
pitch = snd.to_pitch()
f0 = pitch.values()  # Returns numpy array directly
```

### API Summary

| Feature | parselmouth | praatfan |
|---------|-------------|----------|
| Load from file | `Sound("path")` | `Sound("path")` |
| call() function | `from parselmouth.praat import call` | `from praatfan import call` |
| Frame indexing | 1-based | 1-based (in call()) |
| Backend switching | No | Yes |

### Legacy: praatfan_selector

**DEPRECATED:** The `praatfan_selector` package has been merged into `praatfan`. Existing code will continue to work with a deprecation warning:

```python
# Old (deprecated)
from praatfan_selector import Sound, call

# New (preferred)
from praatfan import Sound, call
```

### Supported call() Commands

| Object | Commands |
|--------|----------|
| Sound | `To Pitch (ac)`, `To Pitch (cc)`, `To Formant (burg)`, `To Intensity`, `To Harmonicity (ac/cc)`, `To Spectrum`, `To Spectrogram`, `Get total duration`, `Extract part`, `Filter (pre-emphasis)` |
| Pitch | `Get number of frames`, `Get time from frame number`, `Get value in frame`, `Get value at time`, `Get strength in frame` |
| Formant | `Get number of frames`, `Get time from frame number`, `Get value at time`, `Get bandwidth at time`, `Get number of formants` |
| Intensity | `Get number of frames`, `Get time from frame number`, `Get value in frame`, `Get value at time` |
| Harmonicity | `Get number of frames`, `Get time from frame number`, `Get value in frame`, `Get value at time` |
| Spectrum | `Get centre of gravity`, `Get standard deviation`, `Get skewness`, `Get kurtosis`, `Get band energy` |
| Spectrogram | `Get number of frames`, `Get time from frame number`, `Get power at` |

### Available Backends

praatfan supports four backends:

| Backend | Package | License | Description |
|---------|---------|---------|-------------|
| `praatfan` | Built-in (src/praatfan) | MIT | Pure Python clean-room implementation |
| `praatfan_rust` | `praatfan_rust` (built with maturin) | MIT | Rust implementation with PyO3 bindings |
| `parselmouth` | `praat-parselmouth` | GPL | Python bindings to Praat |
| `praatfan_gpl` | `praatfan_gpl` | MIT | Separate Rust implementation (formerly praatfan-core) |

**Selection priority** (first available wins):
1. `PRAATFAN_BACKEND` environment variable
2. Config file (`./praatfan.toml` or `~/.praatfan/config.toml`)
3. Auto-detect: praatfan_gpl → praatfan_rust → praatfan → parselmouth

**Note:** The Rust wheel (`praatfan_rust`) is a separate package from the Python implementation (`praatfan`). Both can be installed simultaneously.

---

## 📚 Documentation Categories

When implementing, every formula/constant falls into one of three categories:

### ✅ DOCUMENTED
Explicitly stated in papers or Praat manual. Use directly.

Examples:
- HNR formula: `10 × log₁₀(r / (1-r))` (Boersma 1993, Eq. 4)
- Pre-emphasis: `α = exp(-2π × F × Δt)` (Praat manual)
- Window length = 2× the parameter value (Praat manual: "actual length is twice this value")

### 🔬 DETERMINE via Black-Box Testing
Has a specification but not an exact value. Test options against parselmouth.

Examples:
- "Gaussian window with sidelobes below -120 dB" → find the coefficient that achieves this
- "Kaiser-20" → β ≈ 20, verify via FFT of window
- Frame timing (t1) → test left-aligned, centered, right-aligned options
- DC removal method → test weighted vs unweighted mean

### 📚 STANDARD (from cited references)
When Praat cites a reference, standard techniques from that reference are implicitly available.

Examples:
- Root finding via companion matrix eigenvalues (Numerical Recipes Ch. 9.5)
- Root polishing via Newton-Raphson (Numerical Recipes)
- Burg's LPC algorithm (Childers 1978, pp. 252-255)
- Bessel function series expansion (standard numerical analysis)

---

## 🔍 Validation Methodology

### Step 1: Match the Count FIRST

Before comparing values, ensure you produce the **same number of
frames** as parselmouth, and they should have the same duration. 

```
Your implementation: 158 frames
Parselmouth: 159 frames
→ STOP. Fix frame timing before proceeding.
```

If counts don't match, your frame timing formula is wrong. Do not proceed to value comparison.


### Step 2: Verify intermediate values before final outputs

Multi-step algorithms should be debugged layer by layer. When parselmouth exposes intermediate values, verify those BEFORE debugging the final output.

#### Algorithm Dependencies

Implement in dependency order. If algorithm A depends on B, validate B first.

```
Spectrum (foundation) ──┬──► Spectral Moments
                        └──► Spectrogram (windowed Spectrum)

Intensity (independent)

Pitch (foundation) ─────┬──► Harmonicity AC (derived from Pitch AC strength)
                        └──► Harmonicity CC (derived from Pitch CC strength)

Formant (independent, but complex)
```

**Critical:** Harmonicity is computed directly from Pitch correlation strength.
- If your Pitch implementation is correct, Harmonicity follows automatically
- The formula (from Praat manual, Harmonicity.html): `HNR = 10 × log₁₀(r / (1-r))` where r is the normalized autocorrelation at the pitch period
- Do NOT attempt Harmonicity until Pitch is fully validated

#### Verification Order by Algorithm

**Pitch Analysis** (test in this order):
1. Frame count and times (Step 1)
2. Voicing decisions per frame (`Get value in frame` returns 0 for unvoiced)
3. Candidate strengths via `Get strength in frame` - these are correlation values r
4. Final F0 frequencies

**Formant Analysis** (test in this order):
1. Frame count and times (Step 1)
2. Number of formants per frame (`Get number of formants`)
3. Individual formant frequencies F1, F2, F3 separately
4. Individual bandwidths B1, B2, B3 separately

**Harmonicity** (derived from Pitch):
1. Validate Pitch first (same method: AC→AC, CC→CC)
2. HNR is computed from pitch strength r: `10 × log₁₀(r / (1-r))` (Praat manual, Harmonicity.html)
3. If Pitch strengths match, HNR will match automatically

**Spectrum** (test in this order):
1. Number of bins and frequency resolution (dx)
2. Band energies for specific frequency ranges
3. Spectral moments (CoG, std dev, skewness, kurtosis)

**Spectrogram** (windowed Spectrum):
1. Validate Spectrum first (single-frame FFT)
2. Frame count and time/frequency grid dimensions
3. Power values at specific time-frequency points

**Intensity** (relatively simple):
1. Frame count and times (Step 1)
2. Individual frame values in dB

#### Why This Matters

If your F1 values are wrong, the bug could be in:
- Frame timing (wrong samples selected)
- LPC coefficient calculation (Burg algorithm)
- Root finding (companion matrix eigenvalues)
- Root-to-frequency conversion

By checking formant count per frame first, you narrow down where the bug is. If count matches but values don't, the bug is in later stages.

Similarly, if HNR is wrong but Pitch F0 is correct, check if Pitch *strength* values match—the HNR formula is trivial, so the bug is likely in how strength is computed.

### Step 3: Values should match frame by frame, the averages matter less

Precision means **each frame** should match, not just the average.

**✅ Correct validation:**
```
Frame 47 at t=0.125s: your F1=523.4 Hz, parselmouth F1=523.5 Hz → error: 0.1 Hz ✓
Frame 48 at t=0.135s: your F1=531.2 Hz, parselmouth F1=531.1 Hz → error: 0.1 Hz ✓
```

**❌ Wrong validation:**
```
"Average F1 error across all frames is 0.5 Hz"
→ This hides systematic errors! Some frames could be way off while others compensate.
```

### Step 4: Investigate Outliers

If 95% of frames match but 5% have large errors, those outliers reveal edge cases:
- Frames near signal boundaries
- Frames during unvoiced regions
- Frames at phoneme transitions
- Numerical edge cases (division by near-zero, etc.)

These can follow from:

- Window selection
- Normalization
- Quantization
- Sampling
- Other factors

### Target Tolerances

| Algorithm | Target |
|-----------|--------|
| Formant | F1, F2, F3 within 1 Hz |
| Pitch | Within 0.01 Hz |
| Intensity | Within 0.001 dB |
| Spectrum | Relative error < 1e-10 |
| Spectral moments | Exact match |
| HNR | Within 0.01 dB |

---

## 📖 Key Documentation Sources

### Boersma (1993) - Primary source for Pitch and HNR
- Location: `~/Downloads/boersma-pitchtracking.pdf`
- Online: https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf
- Contains: Complete pitch algorithm, HNR formula, sinc interpolation, Viterbi costs, Gaussian window

### Praat Manual
- Location: `/tmp/praat.github.io/docs/manual/`
- Key pages: `Sound__To_Formant__burg____.html`, `Sound__To_Intensity___.html`, etc.
- See `docs/RESOURCES.md` for full list

### Childers (1978) - Burg's Algorithm
- "Modern Spectrum Analysis", IEEE Press, pp. 252-255
- Explicitly cited by Praat manual for formant analysis

### Numerical Recipes - Root Finding
- Press et al., "Numerical Recipes in C", Chapter 9.5
- Explicitly cited by Praat manual

---

## 🛠️ Implementation Order

Start with the most documented algorithms:

1. **Spectrum** - Standard FFT, fully documented
2. **Spectral Moments** - 100% documented formulas
3. **Intensity** - Mostly documented (some black-box testing needed)
4. **Pitch** - Boersma (1993) is comprehensive
5. **Harmonicity** - Derives from pitch
6. **Formant** - Burg LPC + root finding
7. **Spectrogram** - Standard STFT

See `docs/RECIPE.md` for detailed implementation guidance.

---

## 🐍 Python-First Development Approach

For clean-room reimplementation, **develop in Python first**, then port to Rust.

### Why Python First?

1. **Algorithm correctness is the hard part** - Once Python works, Rust port is mechanical
2. **Direct comparison with parselmouth** - Same language, easy to compare intermediate values
3. **Faster iteration** - No compile cycle, interactive debugging, Jupyter notebooks
4. **More experimentation** - Without source code access, you need to probe parselmouth behavior extensively

### Workflow

1. **Python clean-room implementation** using NumPy (no looking at Praat source)
2. **Validate against parselmouth** until frame-by-frame accuracy achieved
3. **Port to Rust** module by module, using Python as reference
4. **Same tests** comparing Rust output vs Python output (should be identical)

### Keep Python "Rust-Shaped"

Write Python that translates easily to Rust:
- Avoid list comprehensions with complex logic
- Use explicit loops where Rust would need them
- Use NumPy arrays (map to `ndarray` in Rust)
- Avoid dynamic typing tricks

### Project Structure

```
src/
├── python/           # Python implementation (develop here first)
│   ├── spectrum.py
│   ├── pitch.py
│   ├── formant.py
│   └── ...
└── rust/             # Rust port (after Python is validated)
    └── (later)
```

---

## 🔧 Development Environment

**Python with parselmouth** (for validation):
```bash
source . .venv
```

**Getting ground truth:**
```python
import parselmouth
from parselmouth.praat import call

snd = parselmouth.Sound("audio.wav")
formant = call(snd, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
n_frames = call(formant, "Get number of frames")

for i in range(1, n_frames + 1):
    t = call(formant, "Get time from frame number", i)
    f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
    print(f"t={t:.3f}: F1={f1:.1f} Hz")
```

See `scripts/README.md` for more validation examples.

---

## 📁 Project Structure

```
praatfan-core-clean/
├── CLAUDE.md              # This file - project instructions
├── USAGE.md               # Usage guide for Python and WASM APIs
├── PLAN.md                # Black-box testing decisions (gitignored)
├── PROGRESS.md            # Session progress notes (gitignored)
├── memory/                # Claude auto-memory (gitignored, syncthinged)
├── docs/
│   ├── RESOURCES.md       # Available documentation sources
│   ├── RECIPE.md          # Step-by-step implementation guide
│   └── API.md             # Target API specification
├── tests/
│   └── fixtures/          # Test audio files (mono WAV only)
├── scripts/
│   ├── README.md          # Validation script instructions
│   └── compare_*.py       # Comparison scripts against parselmouth
├── src/
│   ├── praatfan/          # Python implementation with backend selector
│   │   ├── __init__.py    # Main exports: Sound, call, set_backend, etc.
│   │   ├── selector.py    # Backend detection and unified wrappers
│   │   ├── compatibility.py # Parselmouth call() re-export
│   │   ├── praat.py       # Parselmouth compatibility layer (call() function)
│   │   ├── sound.py       # Sound loading and basic operations
│   │   ├── spectrum.py    # FFT and spectral moments
│   │   ├── intensity.py   # Intensity analysis
│   │   ├── pitch.py       # Pitch detection (AC and CC methods)
│   │   ├── harmonicity.py # HNR (wraps pitch)
│   │   ├── formant.py     # Formant analysis (Burg LPC)
│   │   └── spectrogram.py # STFT spectrogram
│   └── praatfan_selector/ # DEPRECATED: re-exports from praatfan
│       ├── __init__.py    # Deprecation wrapper
│       ├── selector.py    # Re-exports from praatfan.selector
│       └── compatibility.py # Re-exports from praatfan.compatibility
└── rust/                  # Rust implementation (praatfan_rust)
    ├── Cargo.toml         # Dependencies and feature flags (wasm, python)
    ├── Cargo.lock         # Locked dependency versions
    ├── pyproject.toml     # Maturin config for Python builds (praatfan_rust)
    ├── src/
    │   ├── lib.rs         # Library root with re-exports
    │   ├── sound.rs       # Sound type and WAV loading
    │   ├── spectrum.rs    # Single-frame FFT and spectral moments
    │   ├── intensity.rs   # Intensity analysis
    │   ├── pitch.rs       # Pitch detection (AC and CC)
    │   ├── harmonicity.rs # HNR analysis
    │   ├── formant.rs     # Formant analysis (Burg LPC)
    │   ├── spectrogram.rs # STFT spectrogram
    │   ├── error.rs       # Error types
    │   ├── wasm.rs        # WASM bindings (wasm-bindgen)
    │   └── python.rs      # Python bindings (PyO3)
    └── pkg/               # Built WASM package (after wasm-pack build)
```

---

## 📋 Working Files (gitignored)

### PLAN.md
Tracks black-box testing decisions for 🔬 DETERMINE items. Contains:
- Identified discrepancies between RECIPE and implementation
- Testing strategy and phases
- Black-box testing log with dates and results
- Quick reference for debugging specific symptoms

**Update PLAN.md when:**
- Trying a new option for an undocumented constant
- Discovering a discrepancy
- Recording test results

### PROGRESS.md

Detailed session-by-session progress log. This file documents the clean-room implementation journey for potential publication or case study.

**Format for each session:**

```markdown
## Session: YYYY-MM-DD

### Algorithm: [Algorithm Name]

#### Objective
What was the goal for this session?

#### Approach
- Method 1 tried: [description]
- Method 2 tried: [description]

#### Accuracy Assessment

| Metric | Value |
|--------|-------|
| Total frames | N |
| Voiced frames tested | M |
| Frames within tolerance | X |
| Accuracy | X/M (Y%) |
| Mean error | Z |
| Max error | W |
| Outlier count (>threshold) | K |

#### Histogram (when relevant)

```
Error distribution (Hz):
  0.0-0.1:  ████████████████████████ 142 (89.9%)
  0.1-0.5:  ███                       12 (7.6%)
  0.5-1.0:  █                          3 (1.9%)
  1.0+:     ▏                          1 (0.6%)
```

#### Findings
- What worked / what didn't
- Decision points resolved (DPn)
- Remaining issues

#### Next Steps
- What to try next
```

**Guidelines:**

1. **Log every substantive session** - Not just successes, but failed approaches too
2. **Quantify accuracy explicitly** - Use frame counts, percentages, histograms
3. **Document decision points** - When a 🔬 DETERMINE item is resolved, record the winning option
4. **Note what was tried and why** - For publication, the reasoning matters
5. **Include error distributions** - Histograms reveal patterns that averages hide
6. **Track algorithm dependency** - If debugging Harmonicity, note if Pitch was re-validated

### memory/

Claude auto-memory directory. Gitignored but syncthinged across machines. Contains `MEMORY.md` (index) and individual memory files with project state, findings, and release process notes that persist across conversations.

---

## ⚡ Quick Reference: What to Do When Stuck

| Situation | Action |
|-----------|--------|
| Don't know the exact formula | Check `docs/RESOURCES.md` for documentation sources |
| Formula documented but constant unknown | Black-box test: try options, compare to parselmouth |
| Need standard algorithm (FFT, LPC, etc.) | Use cited references (Numerical Recipes, Childers) |
| Frame count doesn't match | Fix frame timing first (test left/center/right alignment) |
| Values don't match but count is right | Compare frame-by-frame, find the outliers |
| Tempted to look at Praat source | **DON'T.** Find it in documentation or determine via testing |

---

## 🚫 Common Mistakes to Avoid

1. **Looking at Praat source code** - Violates clean room principle
2. **Copying from praatfan-core-rs** - Same problem
3. **Using undocumented constants directly** - Must derive via black-box testing
4. **Validating with averages** - Compare individual frames
5. **Proceeding when frame counts don't match** - Fix timing first
6. **Assuming multi-channel works** - We only support mono
7. **Guessing at formulas** - If not documented, test options systematically

---

## 📊 Current Implementation Status

### Completed

| Component | Python | Rust | WASM | PyO3 |
|-----------|--------|------|------|------|
| Sound | ✅ | ✅ | ✅ | ✅ |
| Spectrum | ✅ | ✅ | ✅ | ✅ |
| Spectral Moments | ✅ | ✅ | ✅ | ✅ |
| Intensity | ✅ | ✅ | ✅ | ✅ |
| Pitch (AC) | ✅ | ✅ | ✅ | ✅ |
| Pitch (CC) | ✅ | ✅ | ✅ | ✅ |
| Harmonicity | ✅ | ✅ | ✅ | ✅ |
| Formant | ✅ | ✅ | ✅ | ✅ |
| Spectrogram | ✅ | ✅ | ✅ | ✅ |
| Per-window Spectral | ✅ | - | - | - |

### Per-Window Spectral Methods (New)

The following methods extract spectral features at specific time points:

```python
# Extract a portion of the sound
part = sound.extract_part(0.1, 0.2)

# Get spectrum at a specific time
spectrum = sound.get_spectrum_at_time(0.15, window_length=0.025)

# Batch extraction of spectral moments
times = np.array([0.1, 0.15, 0.2])
moments = sound.get_spectral_moments_at_times(times)
# Returns: {'times', 'center_of_gravity', 'standard_deviation', 'skewness', 'kurtosis'}

# Batch extraction of band energy
energy = sound.get_band_energy_at_times(times, f_min=0, f_max=1000)
```

These work with all backends via praatfan (fallback to iterative calls if backend doesn't have native implementation).

### Key Files

```
rust/
├── Cargo.toml          # Features: wasm, python
├── pyproject.toml      # Maturin config for Python builds
├── src/
│   ├── lib.rs          # Main library with re-exports
│   ├── wasm.rs         # WASM bindings (wasm-bindgen)
│   ├── python.rs       # Python bindings (PyO3, exports as praatfan_rust)
│   ├── sound.rs        # Sound type and WAV loading
│   ├── pitch.rs        # Pitch analysis (AC/CC methods)
│   ├── formant.rs      # Formant analysis (Burg LPC)
│   ├── intensity.rs    # Intensity analysis
│   ├── harmonicity.rs  # HNR analysis
│   ├── spectrum.rs     # Single-frame FFT
│   └── spectrogram.rs  # STFT spectrogram
└── pkg/                # Built WASM package (after wasm-pack build)
```

### Build Commands

```bash
# Python bindings
cd rust
maturin develop --features python    # Install in current venv
maturin build --features python      # Build wheel

# WASM bindings
cd rust
wasm-pack build --target web --features wasm     # For browsers
wasm-pack build --target nodejs --features wasm  # For Node.js

# Rust tests
cargo test
```

**Note:** The WASM build requires `hound` (WAV-only, for in-memory parsing in browsers).
This is an optional dependency enabled automatically by the `wasm` feature flag.

**For rebuilding and uploading release wheels, see `REBUILD.md`.**

### Release Status

**Current version:** v0.1.5 (Pre-release)

This is a **pre-release** - the API is stabilizing but may still change. Use at your own risk in production.

### Pending v0.1.6 (on `main`, unreleased)

After v0.1.5 shipped, `praatfan-gpl==0.1.5` published to PyPI with three
new native methods: `Sound.to_pitch_cc`, `Sound.resample`, and
`Pitch.strengths`. The unified selector on `main` now routes praatfan_gpl
through each — all `hasattr`-guarded so older wheels keep working via
fallbacks (AC pitch with warning, scipy resample, voiced-mask strengths).

End-to-end verified against published `praatfan-gpl==0.1.5` on 2026-04-20:
all four natives (incl. `to_formant_burg_multi` from v0.1.5) exercise via
the unified interface; full 96-test suite passes with zero warnings.

**Pitch threshold kwargs exposed (2026-04-22):** `Sound.to_pitch_ac` and
`Sound.to_pitch_cc` on the unified interface now accept the full Boersma
(1993) tuning set — `voicing_threshold`, `silence_threshold`, `octave_cost`,
`octave_jump_cost`, `voiced_unvoiced_cost` — plumbed through all four
backends. Previously only parselmouth and praatfan_gpl accepted them; the
pure-Python and Rust PyO3 bindings now do too. Rust bindings rebuilt
locally (`maturin develop --features python`), Python bindings are direct
source edits — all ready locally, no wheel upload yet.

**Silence-normalization fix (2026-04-22):** `global_peak` used by Boersma's
silence-detection bonus now uses `np.percentile(|samples|, 99.99)` instead
of `np.max(|samples|)` in both pure-Python (`src/praatfan/pitch.py`) and
Rust (`rust/src/pitch.rs`). Fixes over-silencing on files with a single
loud transient dominating dynamic range (Buckeye s0101a: 234 → 87 frame
disagreements with praatfan_gpl on 0-60s; 328 → 270 on 60-120s). Chosen by
black-box minimizing voicing disagreement vs praatfan_gpl across 4 files;
see `memory/silence_normalization.md` for the comparison table.

**Two-stage resampler (2026-04-26):** Replaced the FFT-based scipy
`signal.resample` (Python) and rustfft (Rust) paths in `_resample` with
a two-stage windowed-sinc resampler matching Praat's `Sound: Resample`.
The hypothesis was raised in the GPL sibling's `docs/TRANSFERABLE_FINDINGS.md`
as a Decision Point with experiment pseudocode; we verified it via blackbox
comparison to `parselmouth.praat.call(snd, "Resample", new, 50)` — no
Praat C++ source examined.

  - Stage 1: FFT brick-wall lowpass at source rate (cutoff at new Nyquist).
    Provides the long ``1/d`` impulse-response tail and silent-region
    Nyquist baseline that Burg LPC depends on for stability.
  - Stage 2: Pure-sinc Hann-windowed interpolation at fractional output
    positions. Kernel zero-crossings at integer input-sample offsets, Hann
    half-width = `precision + 0.5` input samples. No `1/step` normalization
    (stage 1 did the LPF).

Verified vs parselmouth: real-audio mean diff 2.7e-8, p99 1.3e-7 — essentially
bit-exact except at the very last 1-2 boundary samples. Burg parity dropped
F1 mean 6.06 → 3.37 (-44%), F2 mean 14.07 → 4.10 (-71%), F2 p99 280 → 25
(-91%), F3 mean 18.08 → 4.95 (-73%) on the 5-fixture aggregate. Pitch /
Intensity / HNR / Spectrogram unchanged when run on native-rate audio
(verified bit-identical regression check). The Rust port is bit-identical
to Python within machine precision (max diff 2.78e-16). Implementation
details and full investigation log:
- `src/praatfan/formant.py:_resample` (pure-Python)
- `rust/src/formant.rs::resample` (Rust)
- `docs/RESAMPLER_INVESTIGATION.md` (gitignored — full investigation summary)

**Rust free-wins (2026-04-26):** Per `docs/TRANSFERABLE_FINDINGS.md`:
- **faer for eigendecomposition.** `lpc_roots` in `rust/src/formant.rs` now
  uses `faer::Mat::eigenvalues()` instead of `nalgebra::Schur::try_new`.
  Pure Rust, WASM-compatible, ~LAPACK precision; no hand-tuned `max_niter`
  for degenerate companion matrices.
- **Vendored speexdsp smallft FFT (f64).** 4 files at `rust/src/smallft/`
  — an MIT-licensed c2rust translation of Xiph/Vorbis smallft (public-domain
  FFTPACK port; neither file derives from Praat source). Promoted f32→f64
  across 84 sites; fixed an upstream c2rust operator-precedence bug
  (`dradf.rs:15`) and three constants truncated to f32 precision (HSQT2,
  TAUI, TPI). Forward FFT matches numpy.fft.rfft to ~10% of the f64 N·ε
  ceiling; round-trip recovers within 1.5e-15 on pow-2 sizes through 4096.
  Now drives the resampler's stage-1 brick-wall LPF (rustfft retained for
  spectrum/spectrogram/pitch). Caveat: the dradfg general-radix path panics
  at N=4095 — not used by our pow-2 resampler.

**CI: wheels-only workflow (2026-04-26):** Added `.github/workflows/wheels.yml`
that mirrors the release.yml build matrix (Linux x86_64/aarch64, macOS
ARM64, Windows x86_64/ARM64, py3.9-3.12 + WASM + sdist) but stops at the
artifact-upload step. Trigger via `gh workflow run wheels.yml`. Useful for
sanity-checking compilation across all targets without cutting a release.
A `push: branches: [main]` block with `paths:` filter is included
commented-out for future enablement.

Plan: roll all changes (adapter wins + threshold kwargs + silence-normalization
fix + two-stage resampler + Rust free-wins + wheels CI) into the next bump
(~1–2 weeks) rather than chain a v0.1.6 right after v0.1.5. No PyPI action
needed today. Tests: full 112-test suite passes locally (96 prior + 16 new
threshold tests).

**Install from PyPI:**
```bash
pip install praatfan              # Pure Python (all platforms)
pip install praatfan-rust         # Rust backend (optional, faster)
```

- PyPI: [praatfan](https://pypi.org/project/praatfan/0.1.5/) | [praatfan-rust](https://pypi.org/project/praatfan-rust/0.1.5/)
- GitHub: [v0.1.5 release](https://github.com/ucpresearch/praatfan-core-clean/releases/tag/v0.1.5)

Available `praatfan-rust` platforms:
- Linux x86_64: Python 3.9, 3.10, 3.11, 3.12
- Linux aarch64: Python 3.12
- macOS ARM64: Python 3.9, 3.10, 3.11, 3.12
- Windows x86_64: Python 3.9, 3.10, 3.11, 3.12

### Repository

- **Remote:** https://github.com/ucpresearch/praatfan-core-clean
- **Default branch:** main
- **CI/CD:** GitHub Actions builds wheels and WASM package on release creation

### Untracked Files

- `src/praatfan/__pycache__/` - Python cache (gitignored)

---

## 🔄 Parselmouth Compatibility Layer

The `praatfan.praat` module provides a parselmouth-compatible `call()` function, enabling existing parselmouth scripts to work with praatfan objects with minimal changes.

### Usage

```python
# Instead of:
# import parselmouth
# from parselmouth.praat import call

import praatfan
from praatfan.praat import call

# Load sound (use from_file instead of constructor)
sound = praatfan.Sound.from_file("audio.wav")

# All call() commands work the same way
pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
f0 = call(pitch, "Get value at time", 0.5, "Hertz", "Linear")

formant = call(sound, "To Formant (burg)", 0.005, 5, 5500, 0.025, 50)
f1 = call(formant, "Get value at time", 1, 0.5, "Hertz", "Linear")
```

### Supported Commands

#### Sound Commands
| Command | Description |
|---------|-------------|
| `"To Pitch"` / `"To Pitch (ac)"` / `"To Pitch (cc)"` | Create Pitch object |
| `"To Formant (burg)"` | Create Formant object (Burg's LPC) |
| `"To Intensity"` | Create Intensity object |
| `"To Harmonicity (ac)"` / `"To Harmonicity (cc)"` | Create Harmonicity object |
| `"To Spectrum"` | Create Spectrum object (single-frame FFT) |
| `"To Spectrogram"` | Create Spectrogram object |
| `"Get total duration"` | Get sound duration in seconds |
| `"Extract part"` | Extract a portion of the sound |
| `"Filter (pre-emphasis)"` | Apply pre-emphasis filter |

#### Query Commands (Pitch, Formant, Intensity, Harmonicity)
| Command | Description |
|---------|-------------|
| `"Get number of frames"` | Number of analysis frames |
| `"Get time from frame number"` | Time at frame N (1-based index) |
| `"Get value at time"` | Interpolated value at time t |
| `"Get value in frame"` | Value at frame N (1-based index) |

#### Formant-specific
| Command | Description |
|---------|-------------|
| `"Get bandwidth at time"` | Bandwidth at time t |
| `"Get number of formants"` | Number of formants in frame |

#### Pitch-specific
| Command | Description |
|---------|-------------|
| `"Get strength at time"` | Voicing strength at time t |
| `"Get strength in frame"` | Voicing strength at frame N |

#### Spectrum Commands
| Command | Description |
|---------|-------------|
| `"Get centre of gravity"` / `"Get center of gravity"` | Spectral centroid |
| `"Get standard deviation"` | Spectral spread |
| `"Get skewness"` | Spectral asymmetry |
| `"Get kurtosis"` | Spectral peakedness |
| `"Get band energy"` | Energy in frequency range |

#### Spectrogram Commands
| Command | Description |
|---------|-------------|
| `"Get number of frames"` | Number of time frames |
| `"Get number of frequencies"` | Number of frequency bins |
| `"Get time from frame number"` | Time at frame N |
| `"Get power at"` | Power at (time, frequency) |

### Key Differences from Parselmouth

1. **Sound loading**: Use `praatfan.Sound.from_file(path)` instead of `parselmouth.Sound(path)`
2. **Index conversion**: The compatibility layer handles 1-based to 0-based index conversion automatically
3. **Case insensitive**: Commands are matched case-insensitively

### Files

- `src/praatfan/praat.py` - The compatibility layer implementation
- `tests/test_praat_compat.py` - Comprehensive tests (52 tests)

### Example: Migrating from Parselmouth

```python
# Before (parselmouth)
import parselmouth
from parselmouth.praat import call

snd = parselmouth.Sound("audio.wav")
pitch = call(snd, "To Pitch", 0.01, 75, 600)
formant = call(snd, "To Formant (burg)", 0.005, 5, 5500, 0.025, 50)

# After (praatfan)
import praatfan
from praatfan.praat import call

snd = praatfan.Sound.from_file("audio.wav")  # Only this line changes
pitch = call(snd, "To Pitch", 0.01, 75, 600)
formant = call(snd, "To Formant (burg)", 0.005, 5, 5500, 0.025, 50)
```


# A notice about the use of pip

Use `uv pip` for `pip`
