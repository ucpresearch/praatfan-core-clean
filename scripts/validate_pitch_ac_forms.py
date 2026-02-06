#!/usr/bin/env python3
import numpy as np
import os

from praatfan import Sound, call
from praatfan.selector import set_backend
set_backend("praatfan")

import parselmouth
from parselmouth.praat import call as pm_call

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    os.path.join(BASE, "tests/fixtures/one_two_three_four_five.wav"),
    os.path.join(BASE, "tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav"),
]

def get_pitch_data(pitch_obj, call_fn):
    n_frames = int(call_fn(pitch_obj, "Get number of frames"))
    times, f0_values, voiced = [], [], []
    for i in range(1, n_frames + 1):
        t = call_fn(pitch_obj, "Get time from frame number", i)
        f0 = call_fn(pitch_obj, "Get value in frame", i, "Hertz")
        times.append(t)
        if f0 == 0 or (isinstance(f0, float) and np.isnan(f0)):
            f0_values.append(0.0); voiced.append(False)
        else:
            f0_values.append(float(f0)); voiced.append(True)
    return {"n_frames": n_frames, "times": np.array(times), "f0": np.array(f0_values), "voiced": np.array(voiced)}

def compare_pitch(pf_data, pm_data, form_label):
    print()
    print("=" * 70)
    print(f"  {form_label}")
    print("=" * 70)
    pf_n, pm_n = pf_data["n_frames"], pm_data["n_frames"]
    print(f"  Frame count:  praatfan={pf_n}  parselmouth={pm_n}  match={pf_n == pm_n}")
    pf_vn, pm_vn = pf_data["voiced"].sum(), pm_data["voiced"].sum()
    print(f"  Voiced frames: praatfan={pf_vn}  parselmouth={pm_vn}  match={pf_vn == pm_vn}")
    pf_f0v = pf_data["f0"][pf_data["voiced"]]
    pm_f0v = pm_data["f0"][pm_data["voiced"]]
    if len(pf_f0v) > 0 and len(pm_f0v) > 0:
        print(f"  Mean F0 (voiced): praatfan={np.mean(pf_f0v):.2f} Hz  parselmouth={np.mean(pm_f0v):.2f} Hz")
        print(f"  F0 range: praatfan=[{np.min(pf_f0v):.2f}, {np.max(pf_f0v):.2f}]  parselmouth=[{np.min(pm_f0v):.2f}, {np.max(pm_f0v):.2f}]")
    elif len(pf_f0v) == 0 and len(pm_f0v) > 0:
        print(f"  Mean F0: praatfan=NO VOICED  parselmouth={np.mean(pm_f0v):.2f} Hz")
        print(f"  *** WARNING: praatfan NO voiced frames, parselmouth has {len(pm_f0v)} ***")
    n_compare = min(pf_n, pm_n)
    if n_compare == 0: return
    bv = pf_data["voiced"][:n_compare] & pm_data["voiced"][:n_compare]
    bu = ~pf_data["voiced"][:n_compare] & ~pm_data["voiced"][:n_compare]
    agree = (bv | bu).sum()
    print(f"  Voicing agreement: {agree}/{n_compare} ({100*agree/n_compare:.1f}%)")
    print(f"    Both voiced: {bv.sum()}  Both unvoiced: {bu.sum()}  Disagree: {n_compare - agree}")
    if bv.sum() > 0:
        pf_c = pf_data["f0"][:n_compare][bv]; pm_c = pm_data["f0"][:n_compare][bv]
        errors = np.abs(pf_c - pm_c)
        print(f"  F0 comparison ({bv.sum()} both-voiced frames):")
        print(f"    Mean abs error: {np.mean(errors):.4f} Hz  Max: {np.max(errors):.4f} Hz  Median: {np.median(errors):.4f} Hz")
        bins = [0, 0.01, 0.1, 0.5, 1.0, 5.0, float("inf")]
        labels = ["0-0.01", "0.01-0.1", "0.1-0.5", "0.5-1.0", "1.0-5.0", "5.0+"]
        print(f"    Error distribution:")
        for j in range(len(bins)-1):
            cnt = ((errors >= bins[j]) & (errors < bins[j+1])).sum()
            pct = 100*cnt/len(errors) if len(errors) > 0 else 0
            print(f"      {labels[j]:>10s}: {cnt:4d} ({pct:5.1f}%) {chr(35)*int(pct/2)}")
        print(f"    First 10 both-voiced frames:")
        print(f"    {'Frame':>6s} {'Time':>8s} {'PF F0':>10s} {'PM F0':>10s} {'Error':>10s}")
        idxs = np.where(bv)[0]
        for idx in idxs[:10]:
            print(f"    {idx+1:6d} {pm_data['times'][idx]:8.4f} {pf_data['f0'][idx]:10.4f} {pm_data['f0'][idx]:10.4f} {abs(pf_data['f0'][idx]-pm_data['f0'][idx]):10.4f}")
        if np.max(errors) > 0.01:
            print(f"    Worst 5 frames:")
            for wi in np.argsort(errors)[-5:][::-1]:
                idx = idxs[wi]
                print(f"    Frame {idx+1:4d} t={pm_data['times'][idx]:.4f}: pf={pf_data['f0'][idx]:.4f} pm={pm_data['f0'][idx]:.4f} err={errors[wi]:.4f} Hz")

for filepath in FILES:
    fn = os.path.basename(filepath)
    print()
    print("#" * 70)
    print(f"# FILE: {fn}")
    print("#" * 70)
    pf_snd = Sound.from_file(filepath)
    pm_snd = parselmouth.Sound(filepath)
    # Short 3-arg form
    pf_ps = call(pf_snd, "To Pitch (ac)", 0, 75, 600)
    pm_ps = pm_call(pm_snd, "To Pitch (ac)", 0, 75, 600)
    pf_ds = get_pitch_data(pf_ps, call)
    pm_ds = get_pitch_data(pm_ps, pm_call)
    compare_pitch(pf_ds, pm_ds, f"Short form: call(snd, 'To Pitch (ac)', 0, 75, 600) -- {fn}")
    # Full 10-arg form
    pf_pf = call(pf_snd, "To Pitch (ac)", 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
    pm_pf = pm_call(pm_snd, "To Pitch (ac)", 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
    pf_df = get_pitch_data(pf_pf, call)
    pm_df = get_pitch_data(pm_pf, pm_call)
    compare_pitch(pf_df, pm_df, f"Full form: call(snd, 'To Pitch (ac)', 0, 75, 15, 'no', ..., 600) -- {fn}")
    # Cross-form comparison
    print()
    print("=" * 70)
    print(f"  Cross-form comparison: Short vs Full -- {fn}")
    print("=" * 70)
    psv, pfv = pf_ds["voiced"].sum(), pf_df["voiced"].sum()
    print(f"  praatfan  short voiced={psv}  full voiced={pfv}  match={psv == pfv}")
    pmsv, pmfv = pm_ds["voiced"].sum(), pm_df["voiced"].sum()
    print(f"  parselmouth short voiced={pmsv}  full voiced={pmfv}  match={pmsv == pmfv}")
    if pfv < psv * 0.5 and psv > 10:
        print(f"  *** BUG DETECTED: full form has {pfv} voiced vs {psv} short. pitch_ceiling likely set to 15! ***")
    elif pfv == 0 and pmfv > 0:
        print(f"  *** BUG DETECTED: praatfan full form 0 voiced, parselmouth has {pmfv}. pitch_ceiling wrong! ***")
    else:
        print(f"  Both forms produce consistent results -- pitch_ceiling mapping appears correct.")

print()
print("#" * 70)
print("# SUMMARY")
print("#" * 70)
print()
print("If the full 10-arg form works correctly, both forms should produce")
print("similar voiced frame counts and F0 values. If the full form has drastically")
print("fewer voiced frames, pitch_ceiling was incorrectly set to 15 instead of 600.")
print()
