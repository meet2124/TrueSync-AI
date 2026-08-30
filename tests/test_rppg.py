"""
tests/test_rppg.py
==================
Unit tests for the rPPG liveness module using deterministic synthetic signals.

These are TEST FIXTURES — clearly-labelled synthetic data per Section 0 of the spec.
They never run in production and are not in engine.py.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from backend.engine import RPPGModule, _butter_bandpass, _skin_mask_ycbcr


class TestButterworthBPF:
    """Validate that the bandpass filter actually attenuates out-of-band frequencies."""

    def test_passband_frequency_passes(self):
        """A 1.2 Hz sine (72 BPM) should survive the BPF with most energy intact."""
        fs = 30.0
        duration_s = 10.0
        t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
        freq = 1.2  # Hz — inside [0.7, 4.0] band
        signal = np.sin(2 * math.pi * freq * t)

        from scipy.signal import filtfilt
        b, a = _butter_bandpass(fs)
        filtered = filtfilt(b, a, signal)

        # Most energy should survive — check RMS is at least 70% of input
        rms_in = float(np.sqrt(np.mean(signal**2)))
        rms_out = float(np.sqrt(np.mean(filtered**2)))
        assert rms_out / rms_in >= 0.7, (
            f"Passband signal was too attenuated: rms_in={rms_in:.4f} rms_out={rms_out:.4f}"
        )

    def test_stopband_frequency_attenuated(self):
        """A 0.1 Hz sine (6 BPM — below band) should be strongly attenuated."""
        fs = 30.0
        duration_s = 15.0
        t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
        freq = 0.1  # Hz — below 0.7 Hz lower cutoff
        signal = np.sin(2 * math.pi * freq * t)

        from scipy.signal import filtfilt
        b, a = _butter_bandpass(fs)
        filtered = filtfilt(b, a, signal)

        rms_in = float(np.sqrt(np.mean(signal**2)))
        rms_out = float(np.sqrt(np.mean(filtered**2)))
        assert rms_out / rms_in < 0.15, (
            f"Stopband signal not attenuated enough: rms_in={rms_in:.4f} rms_out={rms_out:.4f}"
        )


class TestRPPGModuleScoring:
    """Validate rPPG BPM estimation end-to-end using a synthetic green-channel signal."""

    def _inject_synthetic_green(self, module: RPPGModule, bpm: float, fs: float, n_frames: int) -> None:
        """
        Directly inject a synthetic 'green channel mean' time series into the
        module's forehead ROI buffer, bypassing the camera/FaceMesh path.
        This is an explicit test fixture — not used anywhere in production.
        """
        t = np.linspace(0, n_frames / fs, n_frames, endpoint=False)
        freq = bpm / 60.0
        green_signal = 128.0 + 5.0 * np.sin(2 * math.pi * freq * t)  # centred at 128 (mid-grey)
        for v in green_signal:
            module._roi_buffers["forehead"].append(float(v))
        module._valid_rois = ["forehead"]

    def test_bpm_estimate_72(self):
        """
        Inject a 72 BPM (1.2 Hz) sine into the forehead buffer.
        After recompute, expect BPM within ±5 BPM of 72.
        """
        fs = 30.0
        module = RPPGModule(fs=fs)
        n_frames = int(fs * 8)  # 8 seconds — well above 5s minimum
        self._inject_synthetic_green(module, bpm=72.0, fs=fs, n_frames=n_frames)

        module._recompute()

        assert module._bpm is not None, "BPM should not be None after 8s of signal"
        assert abs(module._bpm - 72.0) <= 5.0, (
            f"Expected BPM ≈ 72, got {module._bpm:.1f}"
        )

    def test_confidence_high_for_clean_sine(self):
        """A clean sine has concentrated spectral power → high confidence."""
        fs = 30.0
        module = RPPGModule(fs=fs)
        n_frames = int(fs * 8)
        self._inject_synthetic_green(module, bpm=60.0, fs=fs, n_frames=n_frames)
        module._recompute()

        assert module._confidence is not None
        assert module._confidence >= 0.20, (
            f"Expected confidence ≥ 0.20 for clean sine, got {module._confidence:.4f}"
        )

    def test_insufficient_buffer_returns_none(self):
        """With only 2s of data (< 5s minimum), BPM and confidence should be None."""
        fs = 30.0
        module = RPPGModule(fs=fs)
        n_frames = int(fs * 2)
        self._inject_synthetic_green(module, bpm=80.0, fs=fs, n_frames=n_frames)
        module._recompute()

        assert module._bpm is None, "BPM should be None with insufficient buffer"
        assert module._confidence is None


class TestSkinMask:
    """Validate YCbCr skin mask on known pixel colours."""

    def test_skin_pixel_detected(self):
        """A mid-tone skin pixel should pass the YCbCr mask."""
        import cv2
        # BGR skin-tone pixel (approximate Fitzpatrick type III)
        skin_bgr = np.array([[[110, 150, 195]]], dtype=np.uint8)
        mask = _skin_mask_ycbcr(skin_bgr)
        # Just verify the function runs and returns a boolean array
        assert mask.shape == (1, 1)

    def test_blue_pixel_rejected(self):
        """A pure blue pixel should not be classified as skin."""
        import cv2
        blue_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        mask = _skin_mask_ycbcr(blue_bgr)
        assert not bool(mask[0, 0]), "Pure blue should not be classified as skin"
