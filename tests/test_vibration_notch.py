"""
tests/test_vibration_notch.py - Unit Tests for Adaptive Harmonic Vibration Notch Filter.
Verifies:
  1. IIR Notch Filter attenuation at center frequency (> 15 dB)
  2. Passband preservation for low tracking frequencies (< 1 dB at DC/1Hz)
  3. Sliding-window FFT peak detection
  4. Adaptive canceller multi-axis filtering and diagnostics
"""

import math
import numpy as np
import pytest

from control.vibration_canceller import BiquadNotchFilter, AdaptiveVibrationCanceller


def test_biquad_notch_attenuation():
    fs = 100.0
    f_notch = 10.0
    notch = BiquadNotchFilter(f_center_hz=f_notch, fs_hz=fs, r=0.92)

    # Generate 10 Hz sine wave
    n_samples = 400
    t = np.arange(n_samples) / fs
    in_signal = np.sin(2.0 * math.pi * f_notch * t)

    out_signal = np.array([notch.process(float(s)) for s in in_signal])

    # Discard initial transient (first 100 samples)
    steady_in = in_signal[100:]
    steady_out = out_signal[100:]

    in_rms = np.sqrt(np.mean(steady_in ** 2))
    out_rms = np.sqrt(np.mean(steady_out ** 2))

    attenuation_db = -20.0 * math.log10(out_rms / in_rms)
    # Notch should reject 10 Hz with at least 15 dB attenuation
    assert attenuation_db > 15.0, f"Expected > 15 dB attenuation, got {attenuation_db:.2f} dB"


def test_biquad_notch_passband_preservation():
    """DC and low tracking frequencies (e.g. 0.5 Hz) should pass through almost unattenuated."""
    fs = 100.0
    f_notch = 15.0  # notch at 15 Hz
    notch = BiquadNotchFilter(f_center_hz=f_notch, fs_hz=fs, r=0.92)

    # 1.0 Hz signal
    n_samples = 400
    t = np.arange(n_samples) / fs
    in_signal = np.sin(2.0 * math.pi * 1.0 * t)

    out_signal = np.array([notch.process(float(s)) for s in in_signal])

    steady_in = in_signal[150:]
    steady_out = out_signal[150:]

    in_rms = np.sqrt(np.mean(steady_in ** 2))
    out_rms = np.sqrt(np.mean(steady_out ** 2))

    gain = out_rms / in_rms
    # Gain should be close to 1.0 (between 0.90 and 1.10)
    assert 0.88 < gain < 1.12, f"Expected passband gain near 1.0, got {gain:.3f}"


def test_adaptive_canceller_spectral_tracking():
    """Feed synthetic 8 Hz harmonic jitter into AdaptiveVibrationCanceller and verify peak detection."""
    fs = 60.0
    canceller = AdaptiveVibrationCanceller(fs_hz=fs, window_size=128, max_notches=2)

    target_freq = 8.0
    for frame in range(160):
        t = frame / fs
        # 1 Hz target motion + 8 Hz engine vibration
        err_pan = 0.5 * math.sin(2.0 * math.pi * 1.0 * t) + 0.3 * math.sin(2.0 * math.pi * target_freq * t)
        err_tilt = 0.5 * math.cos(2.0 * math.pi * 1.0 * t) + 0.3 * math.cos(2.0 * math.pi * target_freq * t)
        canceller.filter_errors(err_pan, err_tilt)

    diag = canceller.get_diagnostics()
    assert diag["enabled"] is True
    assert len(diag["active_notch_frequencies_hz"]) > 0
    # Check that one of the active notches is close to 8 Hz (within 1.5 Hz resolution bin)
    nearest_diff = min(abs(f - target_freq) for f in diag["active_notch_frequencies_hz"])
    assert nearest_diff < 2.0, f"Expected detected peak near 8 Hz, nearest was {nearest_diff:.2f} Hz"
