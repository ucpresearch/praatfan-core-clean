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

## 🔄 Parselmouth Compatibility Layer

The `praatfan_selector` module provides a `call()` function that emulates parselmouth's functional API, allowing existing parselmouth scripts to work with any backend.

### Usage

```python
from praatfan_selector import Sound, call, set_backend

# Optionally choose a specific backend
set_backend("parselmouth")  # or "praatfan", "praatfan-core"

# Load audio
snd = Sound("audio.wav")

# Use parselmouth-style call() syntax
pitch = call(snd, "To Pitch (ac)", 0, 75, 600)
f0 = call(pitch, "Get value in frame", 10, "Hertz")

formant = call(snd, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
f1 = call(formant, "Get value at time", 1, 0.5, "Hertz", "Linear")
```

### Supported Commands

| Object | Commands |
|--------|----------|
| Sound | `To Pitch (ac)`, `To Pitch (cc)`, `To Formant (burg)`, `To Intensity`, `To Harmonicity (ac)`, `To Harmonicity (cc)`, `To Spectrum`, `To Spectrogram`, `Get total duration`, `Extract part`, `Filter (pre-emphasis)` |
| Pitch | `Get number of frames`, `Get time from frame number`, `Get value in frame`, `Get value at time`, `Get strength in frame` |
| Formant | `Get number of frames`, `Get time from frame number`, `Get value at time`, `Get bandwidth at time`, `Get number of formants` |
| Intensity | `Get number of frames`, `Get time from frame number`, `Get value in frame`, `Get value at time` |
| Harmonicity | `Get number of frames`, `Get time from frame number`, `Get value in frame`, `Get value at time` |
| Spectrum | `Get centre of gravity`, `Get standard deviation`, `Get skewness`, `Get kurtosis`, `Get band energy` |
| Spectrogram | `Get number of frames`, `Get time from frame number`, `Get power at` |

### Key Differences from parselmouth

1. **Frame indexing**: Parselmouth uses 1-based indices, praatfan uses 0-based. The `call()` function converts automatically.
2. **Backend flexibility**: Same code works with praatfan (Python), praatfan-core (Rust), or parselmouth.
3. **Return types**: Results are unified wrapper objects with consistent methods like `xs()`, `values()`, `n_frames`.

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
│   ├── praatfan/          # Python implementation (reference)
│   │   ├── __init__.py
│   │   ├── sound.py       # Sound loading and basic operations
│   │   ├── spectrum.py    # FFT and spectral moments
│   │   ├── intensity.py   # Intensity analysis
│   │   ├── pitch.py       # Pitch detection (AC and CC methods)
│   │   ├── harmonicity.py # HNR (wraps pitch)
│   │   ├── formant.py     # Formant analysis (Burg LPC)
│   │   ├── spectrogram.py # STFT spectrogram
│   │   └── praat.py       # Parselmouth compatibility layer (call() function)
│   └── praatfan_selector/ # Backend selector with unified API
│       ├── __init__.py    # Exports Sound, call, set_backend, etc.
│       ├── selector.py    # Backend detection and unified wrappers
│       └── compatibility.py # Parselmouth call() compatibility layer
└── rust/                  # Rust implementation
    ├── Cargo.toml         # Dependencies and feature flags (wasm, python)
    ├── Cargo.lock         # Locked dependency versions
    ├── pyproject.toml     # Maturin config for Python builds
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

### Key Files

```
rust/
├── Cargo.toml          # Features: wasm, python
├── pyproject.toml      # Maturin config for Python builds
├── src/
│   ├── lib.rs          # Main library with re-exports
│   ├── wasm.rs         # WASM bindings (wasm-bindgen)
│   ├── python.rs       # Python bindings (PyO3, parselmouth-compatible)
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

**For rebuilding and uploading release wheels, see `REBUILD.md`.**

### Release Status

**Current version:** v0.1.0 (Pre-release)

This is a **pre-release** - the API is stabilizing but may still change. Use at your own risk in production.

**Pre-release wheels available at:** https://github.com/ucpresearch/praatfan-core-clean/releases/tag/v0.1.0

Available platforms:
- Pure Python: `praatfan-0.1.0-py3-none-any.whl` (works everywhere)
- Linux x86_64: Python 3.9, 3.10, 3.11, 3.12
- macOS ARM64: Python 3.9, 3.10, 3.11, 3.12
- Windows x86_64: Python 3.9, 3.10, 3.11, 3.12

### Repository

- **Remote:** https://github.com/ucpresearch/praatfan-core-clean
- **Default branch:** master
- **CI/CD:** GitHub Actions builds wheels on release creation

### Untracked Files (not committed)

- `rust/examples/dump_f*.rs` - Rust example programs for formant extraction
- `scripts/*.py` - Comparison and debug scripts used during development
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

