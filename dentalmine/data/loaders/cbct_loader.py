"""CBCT volume loader — reads a 3D volume (NIfTI or DICOM series) into axial
slices + physical slice spacing, for the 3D inference path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class LoadedVolume:
    slices: List[np.ndarray]      # list of 2D float32 slices [H, W]
    spacing_mm: float             # physical spacing between slices (z)
    path: str
    shape: tuple                  # (Z, H, W)


def load_cbct(path: str) -> LoadedVolume:
    """Load a .nii/.nii.gz file or a directory holding a DICOM series."""
    import SimpleITK as sitk

    p = Path(path)
    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesFileNames(str(p))
        if not ids:
            raise FileNotFoundError(f"No DICOM series found in {p}")
        reader.SetFileNames(ids)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(str(p))

    spacing = img.GetSpacing()            # (sx, sy, sz)
    z_spacing = float(spacing[2]) if len(spacing) >= 3 else 1.0
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # [Z, H, W]
    slices = [arr[i] for i in range(arr.shape[0])]
    return LoadedVolume(slices=slices, spacing_mm=z_spacing, path=str(p), shape=arr.shape)
