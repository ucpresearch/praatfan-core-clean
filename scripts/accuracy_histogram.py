#!/usr/bin/env python3
"""
Generate accuracy histograms for progress reporting.

This script compares praatfan output against parselmouth ground truth
and generates text-based histograms suitable for PROGRESS.md.

Usage:
    python scripts/accuracy_histogram.py <module> <audio_file> [options]

Examples:
    python scripts/accuracy_histogram.py spectrum tests/fixtures/one_two_three_four_five.wav
    python scripts/accuracy_histogram.py formant tests/fixtures/one_two_three_four_five.wav --formant-number 1

Modules: spectrum, intensity, pitch, formant, harmonicity, spectrogram
"""

import argparse
import sys
import numpy as np

# Histogram configuration
HISTOGRAM_BINS = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, float('inf')]
HISTOGRAM_LABELS = ['0.00-0.01', '0.01-0.1', '0.1-0.5', '0.5-1.0', '1.0-5.0', '5.0-10.0', '10.0+']
BAR_WIDTH = 40


def make_histogram(errors: np.ndarray, bins: list = None, labels: list = None) -> str:
    """Generate a text histogram of error values."""
    if bins is None:
        bins = HISTOGRAM_BINS
    if labels is None:
        labels = HISTOGRAM_LABELS

    # Filter out NaN values
    errors = errors[~np.isnan(errors)]
    if len(errors) == 0:
        return "  (no valid data)\n"

    counts, _ = np.histogram(errors, bins=bins)
    total = len(errors)
    max_count = max(counts) if max(counts) > 0 else 1

    lines = []
    max_label_len = max(len(label) for label in labels)

    for i, (label, count) in enumerate(zip(labels, counts)):
        pct = 100 * count / total if total > 0 else 0
        bar_len = int(BAR_WIDTH * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_len if bar_len > 0 else '▏' if count > 0 else ''
        lines.append(f"  {label:>{max_label_len}}: {bar:<{BAR_WIDTH}} {count:>4} ({pct:5.1f}%)")

    return '\n'.join(lines) + '\n'


def accuracy_summary(errors: np.ndarray, tolerance: float) -> dict:
    """Compute accuracy statistics."""
    errors = errors[~np.isnan(errors)]
    n_total = len(errors)
    if n_total == 0:
        return {
            'total': 0,
            'within_tolerance': 0,
            'accuracy': 0.0,
            'mean_error': float('nan'),
            'max_error': float('nan'),
            'median_error': float('nan'),
        }

    within = np.sum(np.abs(errors) <= tolerance)
    return {
        'total': n_total,
        'within_tolerance': int(within),
        'accuracy': 100 * within / n_total,
        'mean_error': float(np.mean(np.abs(errors))),
        'max_error': float(np.max(np.abs(errors))),
        'median_error': float(np.median(np.abs(errors))),
    }


def format_summary_table(stats: dict, tolerance: float, unit: str = '') -> str:
    """Format accuracy summary as markdown table."""
    unit_str = f" {unit}" if unit else ""
    return f"""| Metric | Value |
|--------|-------|
| Total frames | {stats['total']} |
| Within {tolerance}{unit_str} | {stats['within_tolerance']} |
| Accuracy | {stats['accuracy']:.1f}% |
| Mean error | {stats['mean_error']:.4f}{unit_str} |
| Max error | {stats['max_error']:.4f}{unit_str} |
| Median error | {stats['median_error']:.4f}{unit_str} |
"""


def main():
    parser = argparse.ArgumentParser(description='Generate accuracy histograms for progress reporting')
    parser.add_argument('module', choices=['spectrum', 'intensity', 'pitch', 'formant', 'harmonicity', 'spectrogram'],
                        help='Module to test')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--formant-number', type=int, default=1, help='Formant number (1-5) for formant module')
    parser.add_argument('--method', choices=['ac', 'cc'], default='ac', help='Method for pitch/harmonicity')
    parser.add_argument('--tolerance', type=float, help='Tolerance for accuracy calculation')
    args = parser.parse_args()

    # Import modules
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        print("Error: parselmouth not installed. Activate your parselmouth venv.")
        sys.exit(1)

    try:
        # Import praatfan modules as they become available
        # For now, show placeholder message
        print(f"Module: {args.module}")
        print(f"Audio: {args.audio_file}")
        print()
        print("Accuracy histogram generation will work once modules are implemented.")
        print()
        print("Expected output format:")
        print()
        print("```")
        print(f"Error distribution ({args.module}):")
        print(make_histogram(np.array([0.001, 0.005, 0.01, 0.02, 0.1, 0.15, 0.5, 1.2])))
        print("```")
        print()

        example_stats = accuracy_summary(np.array([0.001, 0.005, 0.01, 0.02, 0.1, 0.15, 0.5, 1.2]), 0.1)
        print(format_summary_table(example_stats, 0.1, 'Hz'))

    except ImportError as e:
        print(f"Note: praatfan module not yet implemented: {e}")
        print("This script will provide accuracy comparisons once the Python implementation is ready.")


if __name__ == '__main__':
    main()
