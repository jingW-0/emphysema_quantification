"""
Emphysema Size Classification via Iterative EDT Thresholding
============================================================

Core idea:
    The EDT value at each voxel = distance to nearest non-emphysema
    boundary in mm. This directly encodes which "size class" a voxel
    belongs to — a voxel with EDT > 7.5mm must be inside a hole of
    at least 15mm diameter.

Algorithm (large → small, same iterative logic as LPF method):
    For each radius threshold (7.5, 3.5, 0.75mm):
        1. core  = (EDT > radius) & remaining_mask
           → selects voxel centers deeper than radius inside emphysema
        2. dilate core by radius
           → recovers full hole region from its center outward
        3. intersect with original emphysema mask
           → preserves emphysema index (EI), same as Oh et al.
        4. subtract from remaining_mask
           → next iteration only processes smaller residual holes
    Remainder → E1 (< 1.5mm diameter)

Advantages over LPF:
    - No sigma estimation parameters (beta0, beta1)
    - Size classification is direct from EDT geometry
    - No Gaussian blur distortion near hole boundaries

Advantages over watershed:
    - No seed detection / h-maxima tuning needed
    - Deterministic, parameter-free size classification

Dependencies:
    pip install numpy scipy SimpleITK
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    distance_transform_edt,
    binary_dilation,
    label,
    generate_binary_structure
)
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LAA_THRESHOLD_HU  = -950
LUNG_THRESHOLD_HU = -400
NOISE_VOXEL_MIN   = 2

# Radius thresholds in mm, large → small
# (half of paper's diameter thresholds: 15, 7, 1.5mm)
# Ψ = {7.5, 3.5, 0.75} mm as stated in paper pseudocode
RADIUS_THRESHOLDS_MM = [7.5, 3.5, 0.75]

# Subgroup names indexed by clusterMap value (1=E1 ... 4=E4)
SUBGROUP_NAMES = {1: "E1(<1.5mm)", 2: "E2(1.5-7mm)",
                  3: "E3(7-15mm)",  4: "E4(>=15mm)"}


# ─────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────

@dataclass
class EmphysemaResult:
    laa_percent:   float = 0.0
    e1_volume_ml:  float = 0.0
    e2_volume_ml:  float = 0.0
    e3_volume_ml:  float = 0.0
    e4_volume_ml:  float = 0.0
    e1_fraction:   float = 0.0
    e2_fraction:   float = 0.0
    e3_fraction:   float = 0.0
    e4_fraction:   float = 0.0
    e1_holes:      int   = 0
    e2_holes:      int   = 0
    e3_holes:      int   = 0
    e4_holes:      int   = 0
    lung_volume_ml: float = 0.0
    cluster_map:   np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            "=== Emphysema EDT Clustering Results ===",
            f"  Lung volume : {self.lung_volume_ml:.1f} mL",
            f"  Total %LAA  : {self.laa_percent:.2f}%\n",
            f"  {'Category':<14} {'Holes':>6}  {'Vol(mL)':>8}  {'%LAA':>6}",
            f"  {'-'*40}",
        ]
        for label, name in SUBGROUP_NAMES.items():
            vol  = getattr(self, f'e{label}_volume_ml')
            pct  = getattr(self, f'e{label}_fraction')
            holes= getattr(self, f'e{label}_holes')
            lines.append(f"  {name:<14} {holes:>6}  {vol:>8.3f}  {pct:>6.3f}%")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# DICOM loading
# ─────────────────────────────────────────────

def load_dicom_series(dicom_dir: str) -> tuple:
    """Load DICOM series. Returns (volume_HU float32, spacing_zyx mm)."""
    reader = sitk.ImageSeriesReader()
    names  = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not names:
        raise FileNotFoundError(f"No DICOM files in: {dicom_dir}")
    reader.SetFileNames(names)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z,Y,X)
    sp = image.GetSpacing()
    spacing_zyx = (sp[2], sp[1], sp[0])
    print(f"[load] shape={volume.shape}, spacing(z,y,x)={spacing_zyx} mm")
    return volume, spacing_zyx


# ─────────────────────────────────────────────
# Lung segmentation
# ─────────────────────────────────────────────

def segment_lung(volume: np.ndarray, spacing_zyx: tuple) -> np.ndarray:
    """Segment lung parenchyma, removing trachea/major airways."""
    binary = volume < LUNG_THRESHOLD_HU
    struct = generate_binary_structure(3, 2)
    labeled_arr, n = label(binary, structure=struct)

    sizes = np.bincount(labeled_arr.ravel())
    sizes[0] = 0
    sorted_labels = np.argsort(sizes)[::-1]

    # Keep 1 or 2 largest components (handles connected lungs in severe emphysema)
    SECOND_LUNG_RATIO = 1 / 3
    keep = [sorted_labels[0]]
    if (n >= 2 and
            sizes[sorted_labels[1]] >= SECOND_LUNG_RATIO * sizes[sorted_labels[0]]):
        keep.append(sorted_labels[1])
        print(f"[segment_lung] Two components "
              f"(ratio={sizes[sorted_labels[1]]/sizes[sorted_labels[0]]:.2f})")
    else:
        print(f"[segment_lung] One component (lungs connected or single lung)")

    lung_mask = np.isin(labeled_arr, keep)

    # Remove trachea: largest LAA-connected component
    airway = (volume < LAA_THRESHOLD_HU) & lung_mask
    dil_iter = max(1, int(round(3.0 / np.mean(spacing_zyx))))
    from scipy.ndimage import binary_dilation as bdil
    airway_dil = bdil(airway, structure=generate_binary_structure(3, 1),
                      iterations=dil_iter)
    aw_labeled, _ = label(airway_dil & lung_mask)
    aw_sizes = np.bincount(aw_labeled.ravel())
    aw_sizes[0] = 0
    trachea = aw_labeled == np.argmax(aw_sizes)
    lung_mask = lung_mask & ~trachea

    print(f"[segment_lung] {lung_mask.sum():,} voxels")
    return lung_mask


# ─────────────────────────────────────────────
# Emphysema mask + noise reduction
# ─────────────────────────────────────────────

def extract_emphysema_mask(volume: np.ndarray,
                            lung_mask: np.ndarray) -> np.ndarray:
    return (volume < LAA_THRESHOLD_HU) & lung_mask


def noise_reduction(emph_mask: np.ndarray,
                    min_voxels: int = NOISE_VOXEL_MIN) -> np.ndarray:
    """Remove connected components smaller than min_voxels (6-connectivity)."""
    struct = generate_binary_structure(3, 1)  # 6-connectivity
    labeled_arr, _ = label(emph_mask, structure=struct)
    sizes = np.bincount(labeled_arr.ravel())
    sizes[0] = 0
    cleaned = sizes[labeled_arr] >= min_voxels
    print(f"[noise_reduction] {cleaned.sum():,} voxels remain")
    return cleaned


# ─────────────────────────────────────────────
# EDT computation
# ─────────────────────────────────────────────

def compute_edt(emph_mask: np.ndarray,
                spacing_zyx: tuple) -> np.ndarray:
    """
    Euclidean Distance Transform on emphysema mask.
    Each voxel value = distance in mm to nearest non-emphysema boundary.
    """
    edt = distance_transform_edt(emph_mask, sampling=spacing_zyx)
    print(f"[EDT] max={edt.max():.2f}mm  "
          f"(largest inscribed sphere radius)")
    return edt.astype(np.float32)


# ─────────────────────────────────────────────
# Core: Iterative EDT-based clustering
# ─────────────────────────────────────────────

def edt_iterative_clustering(
        emph_mask: np.ndarray,
        edt: np.ndarray,
        spacing_zyx: tuple,
        radius_thresholds_mm: list = RADIUS_THRESHOLDS_MM
) -> np.ndarray:
    """
    Iterative EDT-threshold + dilation emphysema size classification.

    For each radius threshold (large → small):
      1. core  = (EDT > radius) & remaining_mask
         Voxels deeper than `radius` mm from boundary must be inside
         a hole of diameter > 2*radius — these are the hole centers.

      2. dilate core by `radius` mm
         Expand centers outward to recover the full hole region.
         Dilation radius = same as EDT threshold, by geometric symmetry:
         the center is `radius` mm from the boundary, so dilating by
         `radius` recovers voxels up to the boundary.

      3. intersect with original emphysema mask
         Confines classification to true emphysema voxels (preserves EI).

      4. subtract from remaining_mask
         Next iteration processes only smaller residual holes.

    Key insight vs LPF:
      - EDT threshold replaces Gaussian skeleton selection
        → no sigma/beta parameters, geometrically exact
      - Dilation radius = EDT threshold (not a separate parameter)
      - Original EDT reused across iterations: a voxel's EDT value
        reflects its position in the original hole geometry, not
        the remaining mask — so reuse is physically correct.

    Parameters
    ----------
    emph_mask           : noise-reduced emphysema mask
    edt                 : precomputed EDT in mm (same shape)
    spacing_zyx         : voxel spacing in mm
    radius_thresholds_mm: list of radii large→small

    Returns
    -------
    cluster_map : int array same shape as emph_mask
                  0=unclassified, 1=E1, 2=E2, 3=E3, 4=E4
    """
    mean_spacing = np.mean(spacing_zyx)
    cluster_map  = np.zeros_like(emph_mask, dtype=np.int32)
    remaining    = emph_mask.copy().astype(bool)

    # Subgroup labels in same order as radius_thresholds (large→small)
    # 7.5mm→E4(4), 3.5mm→E3(3), 0.75mm→E2(2), remainder→E1(1)
    subgroup_labels = [4, 3, 2]

    struct6 = generate_binary_structure(3, 1)  # 6-connectivity for dilation

    for radius_mm, sg_label in zip(radius_thresholds_mm, subgroup_labels):

        # ── Step 1: Threshold EDT on remaining mask ──
        # Voxels with EDT > radius_mm are centers of holes larger than
        # this radius — they are geometrically guaranteed to be inside
        # a hole of diameter > 2*radius_mm
        core = (edt > radius_mm) & remaining

        if not core.any():
            print(f"[clustering] E{sg_label} "
                  f"(radius={radius_mm}mm): 0 voxels (no core found)")
            continue

        # ── Step 2: Dilate core by radius_mm ──
        # Convert radius to voxels for dilation iterations
        dil_iters = max(1, int(round(radius_mm / mean_spacing)))
        dilated = binary_dilation(core,
                                  structure=struct6,
                                  iterations=dil_iters)

        # ── Step 3: Intersect with original emphysema mask ──
        # Also exclude already-classified voxels (first-write wins,
        # consistent with large→small order)
        subgroup_mask = dilated & emph_mask & (cluster_map == 0)

        # ── Step 4: Write label and subtract from remaining ──
        cluster_map[subgroup_mask] = sg_label
        remaining = remaining & ~subgroup_mask

        count = subgroup_mask.sum()
        print(f"[clustering] E{sg_label} "
              f"(radius={radius_mm}mm, dil_iters={dil_iters}): "
              f"{count:,} voxels")

    # ── Remainder = E1 (< 0.75mm radius = < 1.5mm diameter) ──
    e1_mask = remaining & emph_mask & (cluster_map == 0)
    cluster_map[e1_mask] = 1
    print(f"[clustering] E1 (<1.5mm): {e1_mask.sum():,} voxels")

    return cluster_map


# ─────────────────────────────────────────────
# Hole counting per subgroup
# ─────────────────────────────────────────────

def count_holes(cluster_map: np.ndarray) -> dict:
    """
    Count distinct emphysema holes per subgroup via connected components.
    Uses 6-connectivity (face neighbors only).
    """
    struct6 = generate_binary_structure(3, 1)
    hole_counts = {}
    for sg_label in [1, 2, 3, 4]:
        mask = cluster_map == sg_label
        _, n = label(mask, structure=struct6)
        hole_counts[sg_label] = n
    return hole_counts


# ─────────────────────────────────────────────
# Compute indices
# ─────────────────────────────────────────────

def compute_emphysema_indices(cluster_map: np.ndarray,
                               lung_mask: np.ndarray,
                               emph_mask: np.ndarray,
                               spacing_zyx: tuple) -> EmphysemaResult:
    voxel_vol_ml  = np.prod(spacing_zyx) / 1000.0
    lung_vol_ml   = lung_mask.sum() * voxel_vol_ml
    laa_pct       = 100.0 * emph_mask.sum() / max(lung_mask.sum(), 1)
    hole_counts   = count_holes(cluster_map)

    def vol(sg):  return (cluster_map == sg).sum() * voxel_vol_ml
    def frac(sg): return 100.0 * (cluster_map == sg).sum() / max(lung_mask.sum(), 1)

    return EmphysemaResult(
        laa_percent    = laa_pct,
        e1_volume_ml   = vol(1),  e1_fraction = frac(1), e1_holes = hole_counts[1],
        e2_volume_ml   = vol(2),  e2_fraction = frac(2), e2_holes = hole_counts[2],
        e3_volume_ml   = vol(3),  e3_fraction = frac(3), e3_holes = hole_counts[3],
        e4_volume_ml   = vol(4),  e4_fraction = frac(4), e4_holes = hole_counts[4],
        lung_volume_ml = lung_vol_ml,
        cluster_map    = cluster_map
    )


# ─────────────────────────────────────────────
# Optional: collapsibility (Eq. 3, Oh et al.)
# ─────────────────────────────────────────────

def compute_collapsibility(result_ins: EmphysemaResult,
                            result_exp: EmphysemaResult) -> dict:
    def safe(ins, exp):
        return float('nan') if ins == 0 else (1 - exp / ins) * 100.0
    return {
        'C_E1': safe(result_ins.e1_volume_ml, result_exp.e1_volume_ml),
        'C_E2': safe(result_ins.e2_volume_ml, result_exp.e2_volume_ml),
        'C_E3': safe(result_ins.e3_volume_ml, result_exp.e3_volume_ml),
        'C_E4': safe(result_ins.e4_volume_ml, result_exp.e4_volume_ml),
    }


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(dicom_dir: str,
                 dicom_dir_exp: Optional[str] = None) -> EmphysemaResult:
    """
    Full EDT iterative clustering pipeline.

    Parameters
    ----------
    dicom_dir     : inspiratory CT DICOM directory
    dicom_dir_exp : expiratory CT DICOM directory (optional)
    """
    print("\n=== INSPIRATORY CT ===")
    volume, spacing = load_dicom_series(dicom_dir)
    lung_mask       = segment_lung(volume, spacing)
    emph_mask       = extract_emphysema_mask(volume, lung_mask)
    cleaned         = noise_reduction(emph_mask)
    edt             = compute_edt(cleaned, spacing)
    cluster_map     = edt_iterative_clustering(cleaned, edt, spacing)
    result_ins      = compute_emphysema_indices(
                          cluster_map, lung_mask, emph_mask, spacing)
    print("\n" + result_ins.summary())

    if dicom_dir_exp:
        print("\n=== EXPIRATORY CT ===")
        vol_e, sp_e  = load_dicom_series(dicom_dir_exp)
        lung_e       = segment_lung(vol_e, sp_e)
        emph_e       = extract_emphysema_mask(vol_e, lung_e)
        clean_e      = noise_reduction(emph_e)
        edt_e        = compute_edt(clean_e, sp_e)
        cm_e         = edt_iterative_clustering(clean_e, edt_e, sp_e)
        result_exp   = compute_emphysema_indices(cm_e, lung_e, emph_e, sp_e)
        print("\n" + result_exp.summary())

        collapse = compute_collapsibility(result_ins, result_exp)
        print("\n=== Collapsibility (Eq. 3) ===")
        for k, v in collapse.items():
            print(f"  {k}: {v:.1f}%")

    return result_ins


# ─────────────────────────────────────────────
# Synthetic test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from scipy.ndimage import distance_transform_edt as edt_fn

    # Synthetic 64x64x64 volume, spacing 0.75mm isotropic
    W, H, D = 64, 64, 64
    spacing = (0.75, 0.75, 0.75)

    # Two spheres: radius 4mm (~E2, diameter 8mm) and radius 10mm (~E4, diameter 20mm)
    emph_mask = np.zeros((D, H, W), dtype=bool)
    for d in range(D):
        for h in range(H):
            for w in range(W):
                dist1 = np.sqrt(((w-20)*0.75)**2 + ((h-32)*0.75)**2 + ((d-32)*0.75)**2)
                dist2 = np.sqrt(((w-44)*0.75)**2 + ((h-32)*0.75)**2 + ((d-32)*0.75)**2)
                if dist1 <= 4.0 or dist2 <= 10.0:
                    emph_mask[d, h, w] = True

    lung_mask = np.ones((D, H, W), dtype=bool)

    print(f"Synthetic volume: {W}x{H}x{D}, spacing={spacing}")
    print(f"Emphysema voxels: {emph_mask.sum():,}\n")

    # Noise reduction
    cleaned = noise_reduction(emph_mask)

    # EDT
    edt = compute_edt(cleaned, spacing)

    # Clustering
    cluster_map = edt_iterative_clustering(cleaned, edt, spacing)

    # Results
    result = compute_emphysema_indices(cluster_map, lung_mask, emph_mask, spacing)
    print("\n" + result.summary())