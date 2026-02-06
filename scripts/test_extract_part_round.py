"""
Validate the extract_part fix: int() was changed to round() in sound.py and praat.py.

The bug: int() truncates toward zero, so int(0.999999 * 22050) = int(22049.97...) = 22049
but round(0.999999 * 22050) = 22050 which is the correct sample index.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src")

import praatfan
from praatfan.praat import call

print("=" * 70)
print("VALIDATION: extract_part round() vs int() fix")
print("=" * 70)

# Setup: create a Sound with known sample values (sample[i] = i)
sr = 22050
n_total = sr * 2
samples = np.arange(n_total, dtype=np.float64)
snd = praatfan.Sound(samples, sr)

all_pass = True

# Test 1: 0.999999 * 22050 = 22049.97795
print()
print("--- Test 1: sr=22050, extract_part(0, 0.999999) ---")
t_end = 0.999999
product = t_end * sr
int_result = int(product)
round_result = round(product)
print(f"  {t_end} * {sr} = {product}")
print(f"  int()   = {int_result}")
print(f"  round() = {round_result}")
print(f"  Difference = {round_result - int_result} sample(s)")

part = snd.extract_part(0.0, t_end)
vals = part.values() if callable(part.values) else part.values
actual = len(vals)
print(f"  extract_part returned {actual} samples")
if actual == round_result:
    print("  PASS: uses round()")
elif actual == int_result:
    print("  FAIL: still uses int()")
    all_pass = False
else:
    print(f"  UNEXPECTED: {actual}")
    all_pass = False

# Test 2: call() compatibility layer
print()
print("--- Test 2: call() compatibility layer ---")
actual2 = -1
try:
    part2 = call(snd, "Extract part", 0.0, t_end, "rectangular", 1.0, "no")
    vals2 = part2.values() if callable(part2.values) else part2.values
    actual2 = len(vals2)
    print(f"  call() returned {actual2} samples")
    if actual2 == round_result:
        print("  PASS: call() uses round()")
    elif actual2 == int_result:
        print("  FAIL: call() still uses int()")
        all_pass = False
    else:
        print(f"  UNEXPECTED: {actual2}")
        all_pass = False
except Exception as e:
    print(f"  call() error: {e}")
    all_pass = False

# Test 3: Comparison with parselmouth
print()
print("--- Test 3: Comparison with parselmouth ---")
try:
    import parselmouth
    from parselmouth.praat import call as pm_call
    pm_snd = parselmouth.Sound(samples, sr)
    pm_part = pm_call(pm_snd, "Extract part", 0.0, t_end, "rectangular", 1.0, "no")
    pm_vals = pm_part.values[0]
    pm_len = len(pm_vals)
    print(f"  parselmouth: {pm_len} samples")
    print(f"  praatfan:    {actual} samples")
    if pm_len == actual:
        print("  PASS: praatfan matches parselmouth")
    else:
        print(f"  MISMATCH: praatfan={actual}, parselmouth={pm_len}")
        all_pass = False
    n_cmp = min(len(vals), pm_len)
    max_diff = np.max(np.abs(vals[:n_cmp] - pm_vals[:n_cmp]))
    print(f"  Max sample value diff = {max_diff}")
except ImportError:
    print("  parselmouth not available, skipping comparison")
except Exception as e:
    print(f"  parselmouth error: {e}")

# Test 4: sr=44100, same edge case
print()
print("--- Test 4: sr=44100, extract_part(0, 0.999999) ---")
sr4 = 44100
samples4 = np.arange(sr4 * 2, dtype=np.float64)
snd4 = praatfan.Sound(samples4, sr4)
t4 = 0.999999
p4 = t4 * sr4
print(f"  {t4} * {sr4} = {p4}")
print(f"  int()={int(p4)}, round()={round(p4)}")
part4 = snd4.extract_part(0.0, t4)
v4 = part4.values() if callable(part4.values) else part4.values
a4 = len(v4)
print(f"  extract_part returned {a4} samples")
if a4 == round(p4):
    print("  PASS")
elif a4 == int(p4):
    print("  FAIL: int() truncation")
    all_pass = False
else:
    print(f"  UNEXPECTED: {a4}")
    all_pass = False

# Test 5: start_time rounding matters too
print()
print("--- Test 5: start_time rounding, extract_part(0.999999, 1.5) ---")
t_start5 = 0.999999
t_end5 = 1.5
start_round = round(t_start5 * sr)
end_round = round(t_end5 * sr)
start_int = int(t_start5 * sr)
end_int = int(t_end5 * sr)
expected_round = end_round - start_round
expected_int = end_int - start_int

part5 = snd.extract_part(t_start5, t_end5)
v5 = part5.values() if callable(part5.values) else part5.values
a5 = len(v5)
print(f"  start={t_start5}, end={t_end5}")
print(f"  round: start_sample={start_round}, end_sample={end_round}, expected_len={expected_round}")
print(f"  int:   start_sample={start_int}, end_sample={end_int}, expected_len={expected_int}")
print(f"  actual len = {a5}")
if a5 == expected_round:
    print("  PASS: start_time rounding correct")
else:
    print(f"  INFO: got {a5} (expected_round={expected_round}, expected_int={expected_int})")

# Test 6: Verify sample values are correct (not just count)
print()
print("--- Test 6: Verify extracted sample VALUES are correct ---")
last_val = vals[-1]
expected_last = round_result - 1
print(f"  Last extracted sample value = {last_val}")
print(f"  Expected (round-based)      = {expected_last}")
print(f"  Expected (int-based)        = {int_result - 1}")
if last_val == expected_last:
    print("  PASS: correct last sample value")
elif last_val == int_result - 1:
    print("  FAIL: last sample matches int() path")
    all_pass = False
else:
    print(f"  UNEXPECTED: {last_val}")

# Summary
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  The fix changes int() to round() in sample index calculation.")
print(f"  Key example: {t_end}s at sr={sr}Hz")
print(f"    int({product})   = {int_result} (truncates, loses 1 sample)")
print(f"    round({product}) = {round_result} (correct)")
print()
print(f"  Test 1 (extract_part direct): {chr(39)}PASS{chr(39) if actual == round_result else chr(39)}FAIL{chr(39)}")
print(f"  Test 2 (call() layer):        {chr(39)}PASS{chr(39) if actual2 == round_result else chr(39)}FAIL{chr(39)}")
print(f"  Test 4 (sr=44100):            {chr(39)}PASS{chr(39) if a4 == round(p4) else chr(39)}FAIL{chr(39)}")
print(f"  Test 6 (sample values):       {chr(39)}PASS{chr(39) if last_val == expected_last else chr(39)}FAIL{chr(39)}")
print()
if all_pass:
    print("  ALL TESTS PASSED - round() fix is working correctly.")
else:
    print("  SOME TESTS FAILED - check output above for details.")
