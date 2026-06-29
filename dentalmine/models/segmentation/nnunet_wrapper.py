"""nnU-Net v2 wrapper for full 3D CBCT segmentation (DentalSegmentator).

Pretrained model: DentalSegmentator (Zenodo 10829675), labels
  0=bg 1=upper_skull 2=mandible 3=upper_teeth 4=lower_teeth 5=mandibular_canal

`segment()` shells out to nnUNetv2_predict (isolates GPU memory) and is gated on
nnU-Net being installed + the dataset present. `get_tooth_instances()` is pure
numpy/scipy — it splits the teeth labels (3,4) into per-tooth connected components
and assigns each a coarse FDI quadrant/position from its centroid; it accepts a
numpy label volume directly, so it is unit-testable without nnU-Net or SimpleITK.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Union

import numpy as np

UPPER_TEETH_LABEL = 3
LOWER_TEETH_LABEL = 4
MIN_TOOTH_VOXELS = 50           # drop tiny connected-component noise


class NNUNetWrapper:
    def __init__(self, dataset_id: str = "112", config: str = "3d_fullres"):
        self.dataset_id = dataset_id
        self.config = config

    @staticmethod
    def available() -> bool:
        return shutil.which("nnUNetv2_predict") is not None

    def segment(self, volume_path: str, output_dir: str) -> str:
        """Run nnU-Net 3D segmentation; returns the produced mask path.

        Raises a clear error if nnU-Net is not installed (rather than faking it).
        """
        if not self.available():
            raise RuntimeError(
                "nnUNetv2_predict not on PATH. Install nnU-Net v2 + the "
                "DentalSegmentator weights (Zenodo 10829675) to run 3D segmentation."
            )
        in_dir = Path(volume_path)
        in_dir = in_dir if in_dir.is_dir() else in_dir.parent
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out),
             "-d", self.dataset_id, "-c", self.config],
            check=True,
        )
        masks = sorted(out.glob("*.nii.gz"))
        if not masks:
            raise RuntimeError(f"nnU-Net produced no masks in {out}")
        return str(masks[0])

    # ---- pure post-processing (testable) ------------------------------
    @staticmethod
    def get_tooth_instances(seg: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Split teeth labels into per-tooth instances keyed by coarse FDI.

        seg: a [Z, H, W] integer label volume, or a path to one (read via SITK).
        Returns {fdi_str: boolean mask}. FDI quadrant/position is a centroid
        heuristic (TODO: replace with a learned tooth-enumeration head):
          quadrant: upper/lower from label 3/4; left/right from x vs volume centre
          position: 1..8 by anterior->posterior order within the quadrant
        """
        from scipy import ndimage

        if isinstance(seg, str):
            import SimpleITK as sitk
            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg))
        seg = np.asarray(seg)
        cx = seg.shape[2] / 2.0          # x midline (left/right split)
        cy = seg.shape[1] / 2.0          # y midline (anterior/posterior ref)

        instances: Dict[str, np.ndarray] = {}
        # collect (quadrant, anterior_dist, mask) then number within each quadrant
        buckets: Dict[int, list] = {1: [], 2: [], 3: [], 4: []}
        for label, is_upper in ((UPPER_TEETH_LABEL, True), (LOWER_TEETH_LABEL, False)):
            comp, n = ndimage.label(seg == label)
            for c in range(1, n + 1):
                mask = comp == c
                if mask.sum() < MIN_TOOTH_VOXELS:
                    continue
                zc, yc, xc = (float(v) for v in np.argwhere(mask).mean(axis=0))
                is_right = xc < cx
                if is_upper:
                    quadrant = 1 if is_right else 2
                else:
                    quadrant = 4 if is_right else 3
                anterior_dist = abs(yc - cy)   # smaller = more anterior (front)
                buckets[quadrant].append((anterior_dist, mask))

        for quadrant, items in buckets.items():
            items.sort(key=lambda t: t[0])     # anterior -> posterior
            for pos, (_, mask) in enumerate(items, start=1):
                fdi = f"{quadrant}{min(pos, 8)}"
                instances[fdi] = mask
        return instances
