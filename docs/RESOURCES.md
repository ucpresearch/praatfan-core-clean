# Available Documentation Resources

This document lists all non-GPL resources available for clean room implementation.

## Primary Academic Papers

### Boersma (1993) - Pitch and HNR

**Citation:** Boersma, P. (1993). "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound." *Proceedings of the Institute of Phonetic Sciences*, 17, 97-110. University of Amsterdam.

**Location:** `~/Downloads/boersma-pitchtracking.pdf`

**Online:** https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf

**Covers:**
- Complete pitch detection algorithm (autocorrelation method)
- HNR (Harmonics-to-Noise Ratio) calculation
- Windowing (Hanning, Gaussian)
- Sinc interpolation for sub-sample precision (Eq. 22)
- Candidate strength formulas (Eqs. 23, 24)
- Viterbi path finding (Eqs. 26, 27)
- Default parameters

### Childers (1978) - Burg LPC Algorithm

**Citation:** Childers, D.G. (Ed.) (1978). *Modern Spectrum Analysis*. IEEE Press.

**Relevant pages:** 252-255 (Anderson's chapter on Burg's method)

**Covers:**
- Burg's algorithm for LPC coefficient estimation
- Explicitly cited by Praat manual for formant analysis

### Press et al. (1992) - Numerical Recipes

**Citation:** Press, W.H., Teukolsky, S.A., Vetterling, W.T., & Flannery, B.P. (1992). *Numerical Recipes in C: The Art of Scientific Computing* (2nd ed.). Cambridge University Press.

**Relevant chapters:**
- Chapter 9.5: Roots of polynomials (companion matrix, Newton-Raphson polishing)
- Cited by Praat manual for formant analysis

**Note:** Newer editions (3rd ed., 2007) have same algorithms.

### Markel & Gray (1976) - LPC Theory

**Citation:** Markel, J.D. & Gray, A.H. (1976). *Linear Prediction of Speech*. Springer-Verlag.

**Covers:**
- LPC theory and pole-to-formant conversion
- Bandwidth estimation from pole magnitude
- LPC stability (reflecting unstable poles)

## Praat Manual

**Location:** `/tmp/praat.github.io/docs/manual/`

**Online:** https://www.fon.hum.uva.nl/praat/manual/

### Key Algorithm Pages

| Page | Content |
|------|---------|
| `Sound__To_Pitch__ac____.html` | Pitch (AC) parameters |
| `Sound__To_Pitch__raw_autocorrelation____.html` | Raw autocorrelation pitch |
| `Sound__To_Pitch__raw_cross-correlation____.html` | Raw cross-correlation pitch |
| `pitch_analysis_by_raw_autocorrelation.html` | Detailed pitch algorithm |
| `pitch_analysis_by_raw_cross-correlation.html` | CC method details |
| `how_to_choose_a_pitch_analysis_method.html` | AC vs CC comparison |
| `Sound__To_Formant__burg____.html` | Formant analysis |
| `Sound__To_Intensity___.html` | Intensity analysis |
| `Intro_6_2__Configuring_the_intensity_contour.html` | DC removal documentation |
| `Sound__To_Harmonicity__ac____.html` | HNR (AC method) |
| `Sound__To_Harmonicity__cc____.html` | HNR (CC method) |
| `Sound__To_Spectrum___.html` | FFT spectrum |
| `Spectrum__Get_centre_of_gravity___.html` | Spectral moments |
| `Sound__To_Spectrogram___.html` | Time-frequency analysis |
| `Sound__Pre-emphasize__in-place____.html` | Pre-emphasis filter |
| `Sound__Resample___.html` | Resampling algorithm |
| `Sounds__Cross-correlate___.html` | Cross-correlation definition |

### Reading Manual Pages

```bash
# Example: read formant page
cat /tmp/praat.github.io/docs/manual/Sound__To_Formant__burg____.html | \
  sed 's/<[^>]*>//g' | grep -v "^$"
```

Or open in browser: `file:///tmp/praat.github.io/docs/manual/Sound__To_Formant__burg____.html`

## Standard DSP References

These are standard signal processing techniques not specific to Praat:

| Topic | Standard Reference |
|-------|-------------------|
| FFT | Any DSP textbook (Oppenheim & Schafer, etc.) |
| Window functions | Harris (1978) "On the Use of Windows for Harmonic Analysis" |
| Kaiser window | Kaiser & Schafer (1980) |
| Cross-correlation | Standard time series analysis |
| LPC theory | Makhoul (1975) "Linear Prediction: A Tutorial Review" |

## Time Series Analysis

For understanding "forward cross-correlation":

**Positive lag (forward):** Compare signal at time t with signal at time t+τ
**Negative lag (backward):** Compare signal at time t with signal at time t-τ

For pitch/periodicity detection, only positive lags are needed (τ from 1/pitch_ceiling to 1/pitch_floor).

## What These Resources Provide

| Algorithm | Primary Source | Coverage |
|-----------|---------------|----------|
| Pitch (AC) | Boersma (1993) | 95% - comprehensive |
| Pitch (CC) | Manual + Boersma | 70% - window=1 period, forward lags |
| Formant | Manual + Childers + Numerical Recipes | 90% - Burg + root finding |
| Intensity | Manual | 85% - Kaiser-20, DC removal documented |
| Harmonicity (AC) | Boersma (1993) | 90% - derives from pitch |
| Harmonicity (CC) | Manual | 70% - same as pitch CC |
| Spectrum | Manual | 98% - standard FFT |
| Spectral Moments | Manual | 100% - fully documented |
| Spectrogram | Manual | 90% - standard STFT |
