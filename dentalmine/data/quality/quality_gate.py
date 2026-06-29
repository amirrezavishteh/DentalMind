"""Image quality gate — rejects unusable radiographs before inference.

Reject if: SNR < 15 dB OR entropy < 4.5 bits OR >40% pixels outside [10,245].
DICOM tags (modality_hint, slice_spacing, kVp, mAs) are extracted when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import entropy as scipy_entropy


class QualityError(Exception):
    """Raised when an image fails the quality gate."""


@dataclass
class QualityReport:
    snr_db: float
    entropy_bits: float
    overexposed_fraction: float
    passed: bool
    reason: str = ""
    dicom_meta: dict = field(default_factory=dict)


@dataclass
class DicomMeta:
    modality_hint: Optional[str] = None
    slice_spacing: Optional[float] = None
    kvp: Optional[float] = None
    mas: Optional[float] = None


class QualityGate:
    def __init__(
        self,
        min_snr_db: float = 15.0,
        min_entropy_bits: float = 4.5,
        max_overexposed_fraction: float = 0.40,
    ):
        self.min_snr_db = min_snr_db
        self.min_entropy_bits = min_entropy_bits
        self.max_overexposed_fraction = max_overexposed_fraction

    @classmethod
    def from_config(cls, cfg) -> "QualityGate":
        q = cfg.get("quality", {}) if hasattr(cfg, "get") else {}
        return cls(
            min_snr_db=float(q.get("min_snr_db", 15.0)),
            min_entropy_bits=float(q.get("min_entropy_bits", 4.5)),
            max_overexposed_fraction=float(q.get("max_overexposed_fraction", 0.40)),
        )

    # ---- metrics -------------------------------------------------------
    @staticmethod
    def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.ndim == 3:
            img = img.mean(axis=2)
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            if mx > mn:
                img = (img - mn) / (mx - mn) * 255.0
            img = img.astype(np.uint8)
        return img

    @staticmethod
    def estimate_noise_std(gray: np.ndarray) -> float:
        """Immerkaer noise std estimate (Laplacian-mask convolution).

        Robustly isolates sensor noise from anatomical structure, independent
        of image content — far more reliable than corner-patch variance, which
        is corrupted by soft-tissue gradients in real radiographs.
        """
        from scipy.signal import convolve2d

        g = gray.astype(np.float64)
        h, w = g.shape
        mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
        conv = convolve2d(g, mask, mode="valid")
        denom = 6.0 * max((w - 2), 1) * max((h - 2), 1)
        return float(np.sqrt(0.5 * np.pi) * np.abs(conv).sum() / denom)

    @classmethod
    def estimate_snr_db(cls, gray: np.ndarray) -> float:
        """SNR in dB = 20*log10(signal_mean / noise_std).

        Signal = mean intensity of foreground anatomy (pixels above the median);
        noise = Immerkaer estimate.
        """
        g = gray.astype(np.float64)
        noise_std = cls.estimate_noise_std(g) + 1e-6
        med = float(np.median(g))
        fg = g[g > med]
        signal_mean = float(fg.mean()) if fg.size else float(g.mean())
        signal_mean = max(signal_mean, 1e-6)
        return 20.0 * np.log10(signal_mean / noise_std)

    @staticmethod
    def compute_entropy_bits(gray: np.ndarray) -> float:
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        p = hist.astype(np.float64)
        if p.sum() == 0:
            return 0.0
        p = p / p.sum()
        return float(scipy_entropy(p, base=2))

    @staticmethod
    def overexposed_fraction(gray: np.ndarray) -> float:
        outside = (gray < 10) | (gray > 245)
        return float(outside.mean())

    # ---- public API ----------------------------------------------------
    def check(self, image: np.ndarray, dicom_meta: Optional[dict] = None) -> QualityReport:
        gray = self._to_gray_uint8(image)
        snr = self.estimate_snr_db(gray)
        ent = self.compute_entropy_bits(gray)
        over = self.overexposed_fraction(gray)

        reasons = []
        if snr < self.min_snr_db:
            reasons.append(f"SNR {snr:.1f}dB < {self.min_snr_db}dB")
        if ent < self.min_entropy_bits:
            reasons.append(f"entropy {ent:.2f} bits < {self.min_entropy_bits}")
        if over > self.max_overexposed_fraction:
            reasons.append(
                f"{over*100:.0f}% pixels outside [10,245] > "
                f"{self.max_overexposed_fraction*100:.0f}%"
            )
        passed = not reasons
        return QualityReport(
            snr_db=snr,
            entropy_bits=ent,
            overexposed_fraction=over,
            passed=passed,
            reason="; ".join(reasons),
            dicom_meta=dicom_meta or {},
        )

    def check_or_raise(self, image: np.ndarray, dicom_meta: Optional[dict] = None) -> QualityReport:
        report = self.check(image, dicom_meta)
        if not report.passed:
            raise QualityError(f"Image rejected by quality gate: {report.reason}")
        return report

    def check_all(self, images, dicom_meta: Optional[dict] = None):
        """Check a list (or single) of images; raises on first failure."""
        if isinstance(images, np.ndarray) and images.ndim in (2, 3):
            return [self.check_or_raise(images, dicom_meta)]
        return [self.check_or_raise(img, dicom_meta) for img in images]

    @staticmethod
    def read_dicom_meta(dicom_path: str) -> DicomMeta:
        """Extract modality_hint / slice_spacing / kVp / mAs from DICOM tags."""
        try:
            import pydicom
        except ImportError:  # pragma: no cover - optional dep
            return DicomMeta()
        try:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        except Exception:
            return DicomMeta()
        spacing = None
        if "PixelSpacing" in ds:
            try:
                spacing = float(ds.PixelSpacing[0])
            except Exception:
                spacing = None
        if spacing is None and "SliceThickness" in ds:
            try:
                spacing = float(ds.SliceThickness)
            except Exception:
                spacing = None
        return DicomMeta(
            modality_hint=getattr(ds, "Modality", None),
            slice_spacing=spacing,
            kvp=float(ds.KVP) if "KVP" in ds else None,
            mas=float(ds.XRayTubeCurrent) if "XRayTubeCurrent" in ds else None,
        )
