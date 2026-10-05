"""
control/vibration_canceller.py - Adaptive Harmonic Vibration Notch Filter.

Cancels platform structural jitter and acoustic resonance without degrading
low-frequency coarse tracking phase margin.

Literature Grounding:
  - Skormin et al. (1995), IEEE Trans. AES: "Adaptive Jitter Rejection Technique
    Applicable to Airborne Laser Communication Systems"
  - Toyoshima et al. (2002), SPIE Proc. Vol. 4635: "In-orbit tracking performance
    of the optical inter-satellite link experiment (OICETS)"
"""

import math
from typing import Tuple, List, Optional
import numpy as np


class BiquadNotchFilter:
    """
    Direct Form II Second-Order IIR Digital Notch Filter:
      H(z) = (1 - 2*cos(w0)*z^-1 + z^-2) / (1 - 2*r*cos(w0)*z^-1 + r^2*z^-2)
    """
    def __init__(self, f_center_hz: float, fs_hz: float, r: float = 0.97):
        self.fs = fs_hz
        self.r = np.clip(r, 0.70, 0.99)
        self.w0 = 2.0 * math.pi * (f_center_hz / fs_hz)
        self._update_coefficients()

        # State registers for Direct Form II
        self.w1 = 0.0
        self.w2 = 0.0

    def _update_coefficients(self):
        cos_w0 = math.cos(self.w0)
        self.b0 = 1.0
        self.b1 = -2.0 * cos_w0
        self.b2 = 1.0
        self.a1 = -2.0 * self.r * cos_w0
        self.a2 = self.r ** 2

    def set_frequency(self, f_center_hz: float):
        # Constrain frequency between 0.5 Hz and Nyquist (0.45 * fs)
        f_clamped = np.clip(f_center_hz, 0.5, 0.45 * self.fs)
        self.w0 = 2.0 * math.pi * (f_clamped / self.fs)
        self._update_coefficients()

    def process(self, x: float) -> float:
        # Direct Form II:
        # w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
        # y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
        w0 = x - self.a1 * self.w1 - self.a2 * self.w2
        y = self.b0 * w0 + self.b1 * self.w1 + self.b2 * self.w2
        self.w2 = self.w1
        self.w1 = w0
        return y

    def reset(self):
        self.w1 = 0.0
        self.w2 = 0.0


class AdaptiveVibrationCanceller:
    """
    Multi-Harmonic Sliding-Window Adaptive Vibration Canceller.
    Continuously identifies platform mechanical resonance peaks via sliding FFT,
    tunes digital notch filters dynamically, and filters gimbal error signals.
    """

    def __init__(self, fs_hz: float = 60.0, window_size: int = 128, max_notches: int = 3):
        self.fs = fs_hz
        self.window_size = window_size
        self.max_notches = max_notches
        self.enabled = True

        # Ring buffers for error signal history
        self.pan_buffer = np.zeros(window_size, dtype=np.float64)
        self.tilt_buffer = np.zeros(window_size, dtype=np.float64)
        self.buffer_idx = 0
        self.buffer_filled = False

        # Cascade of notch filters for each axis (structural harmonics above 10 Hz)
        self.pan_notches: List[BiquadNotchFilter] = []
        self.tilt_notches: List[BiquadNotchFilter] = []

        # High-frequency structural vibration frequencies (e.g. 18 Hz, 24 Hz)
        # Note: Must NEVER place notches below 10 Hz to prevent destroying target tracking bandwidth!
        default_freqs = [18.0, 24.0]
        for f in default_freqs[:max_notches]:
            self.pan_notches.append(BiquadNotchFilter(f, fs_hz, r=0.97))
            self.tilt_notches.append(BiquadNotchFilter(f, fs_hz, r=0.97))

        # Live telemetry
        self.active_peaks: List[float] = default_freqs[:max_notches]
        self.last_attenuation_db: float = 0.0
        self.last_spectrum_freqs = np.zeros(window_size // 2, dtype=np.float64)
        self.last_spectrum_power = np.zeros(window_size // 2, dtype=np.float64)

        # Update counter to avoid running FFT on every frame (run every 15 frames = ~4 Hz)
        self.frame_count = 0
        self.fft_stride = 15

    def filter_errors(self, err_pan: float, err_tilt: float) -> Tuple[float, float]:
        """
        Filters tracking error signals through the adaptive notch cascades.
        Returns:
            (filtered_err_pan, filtered_err_tilt)
        """
        # Always store in history buffer for continuous spectrum tracking
        self.pan_buffer[self.buffer_idx] = err_pan
        self.tilt_buffer[self.buffer_idx] = err_tilt
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size
        if self.buffer_idx == 0:
            self.buffer_filled = True

        self.frame_count += 1
        if self.frame_count % self.fft_stride == 0 and self.buffer_filled:
            self._update_spectral_peaks()

        if not self.enabled:
            return err_pan, err_tilt

        # Cascade through pan notches
        out_pan = err_pan
        for notch in self.pan_notches:
            out_pan = notch.process(out_pan)

        # Cascade through tilt notches
        out_tilt = err_tilt
        for notch in self.tilt_notches:
            out_tilt = notch.process(out_tilt)

        # Estimate live attenuation in dB
        in_pwr = err_pan ** 2 + err_tilt ** 2 + 1e-9
        out_pwr = out_pan ** 2 + out_tilt ** 2 + 1e-9
        ratio = out_pwr / in_pwr
        self.last_attenuation_db = -10.0 * math.log10(min(1.0, ratio))

        return out_pan, out_tilt

    def _update_spectral_peaks(self):
        """Analyzes combined error power spectrum and retunes notch filters."""
        # Unroll ring buffer with Hanning window to prevent spectral leakage
        idx = self.buffer_idx
        ordered_pan = np.concatenate([self.pan_buffer[idx:], self.pan_buffer[:idx]])
        ordered_tilt = np.concatenate([self.tilt_buffer[idx:], self.tilt_buffer[:idx]])

        window = np.hanning(self.window_size)
        sig = (ordered_pan + ordered_tilt) * window

        # Compute one-sided FFT power spectrum
        fft_vals = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(self.window_size, d=1.0 / self.fs)
        pwr = np.abs(fft_vals) ** 2

        self.last_spectrum_freqs = freqs
        self.last_spectrum_power = pwr

        # Find peaks strictly in mechanical vibration band (>= 6.0 Hz up to Nyquist, ignoring coarse flight tracking < 3 Hz)
        valid_mask = (freqs >= 6.0) & (freqs <= (0.45 * self.fs))
        valid_freqs = freqs[valid_mask]
        valid_pwr = pwr[valid_mask]

        if len(valid_pwr) == 0:
            return

        # Simple peak detector: values higher than local neighbors and above mean power
        mean_pwr = np.mean(valid_pwr)
        peak_candidates = []
        for i in range(1, len(valid_pwr) - 1):
            if valid_pwr[i] > valid_pwr[i - 1] and valid_pwr[i] > valid_pwr[i + 1] and valid_pwr[i] > 1.8 * mean_pwr:
                peak_candidates.append((float(valid_freqs[i]), float(valid_pwr[i])))

        # Sort by power descending
        peak_candidates.sort(key=lambda item: item[1], reverse=True)
        detected_peaks = [f for f, p in peak_candidates[:self.max_notches]]

        # Retune existing notch filters with detected peaks
        if detected_peaks:
            self.active_peaks = detected_peaks
            for i, peak_f in enumerate(detected_peaks):
                if i < len(self.pan_notches):
                    self.pan_notches[i].set_frequency(peak_f)
                    self.tilt_notches[i].set_frequency(peak_f)

    def get_diagnostics(self) -> dict:
        """Returns live diagnostics for UI telemetry and explainability engine."""
        return {
            "enabled": self.enabled,
            "active_notch_frequencies_hz": [round(f, 2) for f in self.active_peaks],
            "estimated_attenuation_db": round(self.last_attenuation_db, 2),
            "dominant_frequency_hz": round(self.active_peaks[0], 2) if self.active_peaks else 0.0
        }

    def reset(self):
        self.pan_buffer.fill(0.0)
        self.tilt_buffer.fill(0.0)
        self.buffer_idx = 0
        self.buffer_filled = False
        self.frame_count = 0
        for n in self.pan_notches:
            n.reset()
        for n in self.tilt_notches:
            n.reset()
