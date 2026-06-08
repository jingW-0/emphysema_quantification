"""
Emphysema Size Classification Pipeline
Based on: Oh et al. (2017) - "Size variation and collapse of emphysema holes
at inspiration and expiration CT scan: evaluation with modified length scale
method and image co-registration"
International Journal of COPD 2017:12 2043-2057

Pipeline:
    1. Load DICOM series
    2. Lung segmentation (threshold + morphological ops)
    3. Emphysema mask extraction (<-950 HU)
    4. Noise reduction (remove clusters < 2 voxels)
    5. Iterative Gaussian LPF size-based clustering (large -> small)
    6. Classify into E1(<1.5mm), E2(<7mm), E3(<15mm), E4(>=15mm)
    7. Compute emphysema indices and correlate with PFT (optional)
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    binary_erosion,
    label,
    generate_binary_structure
)
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ─────────────────────────────────────────────
# Constants (from paper, anatomically grounded)
# ─────────────────────────────────────────────
LAA_THRESHOLD_HU = -950          # Low attenuation area threshold
LUNG_THRESHOLD_HU = -400         # Initial lung segmentation threshold
NOISE_VOXEL_MIN = 2              # Remove clusters smaller than this

# Size thresholds in mm (paper Section: Gaussian low-pass filtering)
# E1 < 1.5 mm  : alveolus / noise
# E2 1.5-7 mm  : subacinus
# E3 7-15 mm   : acinus / sublobular
# E4 >= 15 mm  : extra-lobular
SIZE_THRESHOLDS_MM = [15.0, 7.0, 1.5]   # diameter thresholds, large -> small

# Gaussian kernel sigma estimation parameters (Eq. 1 in paper)
# sigma = beta0 + 2 * beta1 * gamma  where gamma = sphere radius
BETA0 = 0.147
BETA1 = 0.1038


# ─────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────
@dataclass
class EmphysemaResult:
    """Holds per-subgroup volumes and fractions."""
    laa_percent: float = 0.0        # Total %LAA

    e1_volume_ml: float = 0.0       # <1.5 mm
    e2_volume_ml: float = 0.0       # 1.5-7 mm
    e3_volume_ml: float = 0.0       # 7-15 mm
    e4_volume_ml: float = 0.0       # >=15 mm

    e1_fraction: float = 0.0        # as % of total lung volume
    e2_fraction: float = 0.0
    e3_fraction: float = 0.0
    e4_fraction: float = 0.0

    voxel_size_ml: float = 0.0      # ml per voxel
    lung_volume_ml: float = 0.0

    # Raw masks for downstream use (e.g. co-registration)
    masks: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Emphysema Size Classification Results ===",
            f"  Lung volume       : {self.lung_volume_ml:.1f} mL",
            f"  Total %LAA        : {self.laa_percent:.2f}%",
            f"  E1 (<1.5 mm)      : {self.e1_fraction:.2f}%  ({self.e1_volume_ml:.1f} mL)",
            f"  E2 (1.5-7 mm)     : {self.e2_fraction:.2f}%  ({self.e2_volume_ml:.1f} mL)",
            f"  E3 (7-15 mm)      : {self.e3_fraction:.2f}%  ({self.e3_volume_ml:.1f} mL)",
            f"  E4 (>=15 mm)      : {self.e4_fraction:.2f}%  ({self.e4_volume_ml:.1f} mL)",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Step 1: DICOM Loading
# ─────────────────────────────────────────────
def load_dicom_series(dicom_dir: str) -> tuple[np.ndarray, tuple[float, ...]]:
    """
    Load a DICOM series from a directory.

    Returns
    -------
    volume : np.ndarray, shape (Z, Y, X), dtype float32, values in HU
    spacing : (z_mm, y_mm, x_mm) voxel spacing in mm
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM files found in: {dicom_dir}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z, Y, X)
    spacing = image.GetSpacing()                                # (x, y, z) in ITK
    spacing_zyx = (spacing[2], spacing[1], spacing[0])         # reorder to (Z,Y,X)

    print(f"[load] Volume shape: {volume.shape}, spacing (z,y,x): {spacing_zyx} mm")
    return volume, spacing_zyx


# ─────────────────────────────────────────────
# Step 2: Lung Segmentation
# ─────────────────────────────────────────────
def segment_lung(volume: np.ndarray, spacing_zyx: tuple) -> np.ndarray:
    """Segment lung parenchyma, removing trachea/major airways."""
    binary = volume < LUNG_THRESHOLD_HU
    # --- Label connected components, keep 1 or 2 largest (lungs) ---
    # Handles two cases:
    #   - Normal: left and right lung separate → two large components
    #   - Severe emphysema: lungs connected through destroyed parenchyma
    #     → appears as one large component
    # Decision rule: keep 2nd largest only if it is >= 1/3 the size of
    # the largest — true contralateral lung will satisfy this; trachea,
    # bowel gas, or other incidental structures will not.
    SECOND_LUNG_RATIO = 1 / 3
 
    struct = generate_binary_structure(3, 2)
    labeled, n_components = label(binary, structure=struct)
 
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    sorted_labels = np.argsort(sizes)[::-1]  # descending by size
 
    largest_size = sizes[sorted_labels[0]]
    keep_labels = [sorted_labels[0]]         # always keep largest
 
    if (n_components >= 2 and
            sizes[sorted_labels[1]] >= SECOND_LUNG_RATIO * largest_size):
        keep_labels.append(sorted_labels[1])
        print(f"[segment_lung] Two lung components "
              f"(size ratio={sizes[sorted_labels[1]]/largest_size:.2f})")
    else:
        print(f"[segment_lung] One lung component "
              f"(connected or single lung)")
 
    lung_mask = np.isin(labeled, keep_labels)
 
    # Remove trachea: largest LAA-connected component
    airway = (volume < LAA_THRESHOLD_HU) & lung_mask
    dil_iter = max(1, int(round(3.0 / np.mean(spacing_zyx))))
    from scipy.ndimage import binary_dilation
    airway_dil = binary_dilation(airway, structure=generate_binary_structure(3,1),
                                  iterations=dil_iter)
    aw_labeled, _ = label(airway_dil & lung_mask)
    aw_sizes = np.bincount(aw_labeled.ravel())
    aw_sizes[0] = 0
    trachea = aw_labeled == np.argmax(aw_sizes)
    lung_mask = lung_mask & ~trachea
 
    print(f"[segment_lung] {lung_mask.sum():,} voxels")
    return lung_mask


# ─────────────────────────────────────────────
# Step 3: Emphysema Mask
# ─────────────────────────────────────────────
def extract_emphysema_mask(volume: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    """
    Extract low-attenuation area (LAA) mask within the lung.
    Threshold: < -950 HU (standard emphysema index threshold).
    """
    emph_mask = (volume < LAA_THRESHOLD_HU) & lung_mask
    print(f"[emphysema_mask] Emphysema voxels: {emph_mask.sum():,}")
    return emph_mask


# ─────────────────────────────────────────────
# Step 4: Noise Reduction
# ─────────────────────────────────────────────
def noise_reduction(emph_mask: np.ndarray, min_voxels: int = NOISE_VOXEL_MIN) -> np.ndarray:
    """
    Remove isolated emphysema clusters smaller than min_voxels.

    This implements the block-matching mode filter described in
    Equations S1-S2 of the paper's supplementary material,
    approximated here as connected-component size filtering
    (equivalent outcome: remove < 2-voxel clusters).
    """
    struct = generate_binary_structure(3, 1)
    labeled, _ = label(emph_mask, structure=struct)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    # Keep only components >= min_voxels
    keep = sizes >= min_voxels
    cleaned = keep[labeled]
    print(f"[noise_reduction] Voxels after cleaning: {cleaned.sum():,}")
    return cleaned


# ─────────────────────────────────────────────
# Step 5: Gaussian Sigma Estimation (Eq. 1)
# ─────────────────────────────────────────────
def estimate_sigma(radius_mm: float) -> float:
    """
    Estimate Gaussian kernel sigma from sphere radius.
    Equation 1: sigma = beta0 + 2 * beta1 * gamma
    where gamma = radius of sphere in mm.
    """
    return BETA0 + 2 * BETA1 * radius_mm


def visualize_lpf_iteration(
    volume: np.ndarray,
    filtered: np.ndarray,
    skeleton_mask: np.ndarray,
    subgroup_mask: np.ndarray,
    radius_mm: float,
    quartiles: tuple[float, float, float] = (0.25, 0.5, 0.75),
    figsize: tuple = (18, 10)
) -> None:
    """
    Visualize the LPF effect and the resulting subgroup mask for one iteration.

    Top row: filtered LPF images with skeleton overlay.
    Bottom row: original axial CT slices with the new subgroup mask overlay.
    """
    z_dim = volume.shape[0]
    z_indices = [max(0, min(z_dim - 1, int(round(q * (z_dim - 1)))))
                 for q in quartiles]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"LPF iteration radius={radius_mm}mm", fontsize=16, fontweight='bold')

    filt_vmin = np.percentile(filtered, 2)
    filt_vmax = np.percentile(filtered, 98)
    img_vmin = np.percentile(volume, 2)
    img_vmax = np.percentile(volume, 98)

    for col, z in enumerate(z_indices):
        # filtered LPF image + skeleton overlay
        ax_f = axes[0, col]
        ax_f.imshow(filtered[z, :, :], cmap='magma', origin='lower', vmin=filt_vmin, vmax=filt_vmax)
        ax_f.contour(skeleton_mask[z, :, :], colors='cyan', linewidths=0.5)
        ax_f.set_title(f"Filtered Z={z}")
        ax_f.axis('off')

        # original image + subgroup overlay
        ax_i = axes[1, col]
        ax_i.imshow(volume[z, :, :], cmap='gray', origin='lower', vmin=img_vmin, vmax=img_vmax)
        ax_i.contour(subgroup_mask[z, :, :], colors='yellow', linewidths=0.5)
        ax_i.set_title(f"Subgroup Z={z}")
        ax_i.axis('off')

    axes[0, 0].set_ylabel('Filtered LPF', fontsize=12)
    axes[1, 0].set_ylabel('Subgroup mask', fontsize=12)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Step 6: Iterative Size-Based Clustering
# ─────────────────────────────────────────────
def size_based_emphysema_clustering(
    volume: np.ndarray,
    noise_reduced_mask: np.ndarray,
    emph_mask: np.ndarray,
    spacing_zyx: tuple,
    size_thresholds_mm: list = SIZE_THRESHOLDS_MM
) -> dict:
    """
    Iterative Gaussian LPF size-based emphysema classification.

    Implements Figure 1 / pseudocode from supplementary material.
    Processes from LARGE to SMALL kernel sizes.

    Parameters
    ----------
    noise_reduced_mask : initial noise-cleaned emphysema mask (Pm,0)
    emph_mask          : original emphysema mask (Em) — used for intersection
    spacing_zyx        : voxel spacing in mm (z, y, x)
    size_thresholds_mm : diameter thresholds in mm, ordered large -> small

    Returns
    -------
    subgroup_masks : dict with keys 'E4', 'E3', 'E2', 'E1', 'remainder'
    """
    subgroup_masks = {}
    current_mask = noise_reduced_mask.copy().astype(np.float32)
    mean_spacing = np.mean(spacing_zyx)

    # Diameter thresholds map to subgroup labels (paper convention)
    threshold_to_label = {15.0: 'E4', 7.0: 'E3', 1.5: 'E2'}

    for diameter_mm in size_thresholds_mm:
        label_name = threshold_to_label[diameter_mm]
        radius_mm = diameter_mm / 2.0

        # Sigma in mm -> convert to voxels per axis
        sigma_mm = estimate_sigma(radius_mm)
        sigma_vox = tuple(sigma_mm / s for s in spacing_zyx)

        # --- Gaussian LPF (Eq. S3-S5) ---
        # Apply to current remaining mask
        filtered = gaussian_filter(current_mask.astype(np.float32), sigma=sigma_vox)

        # --- Select skeleton voxels (Eq. S6) ---
        # Voxels where filtered value equals the global maximum density
        # In practice: voxels at local maxima of the filtered image
        max_val = filtered.max()
        if max_val == 0:
            subgroup_masks[label_name] = np.zeros_like(current_mask, dtype=bool)
            continue

        skeleton_mask = (filtered >= max_val * 0.999).astype(bool)

        # --- Dilation by radius ---
        dilation_voxels = max(1, int(round(radius_mm / mean_spacing)))
        struct = generate_binary_structure(3, 1)
        dilated_mask = binary_dilation(
            skeleton_mask,
            structure=struct,
            iterations=dilation_voxels
        )

        # --- Intersect with original emphysema mask (preserve EI) ---
        size_specific_mask = dilated_mask & emph_mask.astype(bool)

        # Visualize LPF effect and subgroup formation for this iteration
        visualize_lpf_iteration(
            volume,
            filtered,
            skeleton_mask,
            size_specific_mask,
            radius_mm
        )

        # --- Store this subgroup ---
        subgroup_masks[label_name] = size_specific_mask

        # --- Subtract from current mask for next iteration ---
        current_mask = current_mask * (~size_specific_mask).astype(np.float32)

        n_vox = size_specific_mask.sum()
        print(f"[clustering] {label_name} (radius={radius_mm}mm): {n_vox:,} voxels")

    # Remainder = E1 (smallest, < 1.5mm)
    subgroup_masks['E1'] = current_mask.astype(bool) & emph_mask.astype(bool)
    print(f"[clustering] E1 (<1.5mm): {subgroup_masks['E1'].sum():,} voxels")

    return subgroup_masks


# ─────────────────────────────────────────────
# Step 7: Compute Indices
# ─────────────────────────────────────────────
def compute_emphysema_indices(
    subgroup_masks: dict,
    lung_mask: np.ndarray,
    emph_mask: np.ndarray,
    spacing_zyx: tuple
) -> EmphysemaResult:
    """
    Compute volumetric emphysema indices from subgroup masks.
    Matches Table 4 in the paper.
    """
    voxel_vol_mm3 = np.prod(spacing_zyx)
    voxel_vol_ml = voxel_vol_mm3 / 1000.0

    lung_vol_ml = lung_mask.sum() * voxel_vol_ml
    laa_percent = 100.0 * emph_mask.sum() / max(lung_mask.sum(), 1)

    def vol_and_frac(mask):
        vol = mask.sum() * voxel_vol_ml
        frac = 100.0 * mask.sum() / max(lung_mask.sum(), 1)
        return vol, frac

    e1_vol, e1_frac = vol_and_frac(subgroup_masks.get('E1', np.zeros_like(lung_mask)))
    e2_vol, e2_frac = vol_and_frac(subgroup_masks.get('E2', np.zeros_like(lung_mask)))
    e3_vol, e3_frac = vol_and_frac(subgroup_masks.get('E3', np.zeros_like(lung_mask)))
    e4_vol, e4_frac = vol_and_frac(subgroup_masks.get('E4', np.zeros_like(lung_mask)))

    result = EmphysemaResult(
        laa_percent=laa_percent,
        e1_volume_ml=e1_vol, e1_fraction=e1_frac,
        e2_volume_ml=e2_vol, e2_fraction=e2_frac,
        e3_volume_ml=e3_vol, e3_fraction=e3_frac,
        e4_volume_ml=e4_vol, e4_fraction=e4_frac,
        voxel_size_ml=voxel_vol_ml,
        lung_volume_ml=lung_vol_ml,
        masks=subgroup_masks
    )
    return result


def visualize_subgroup_clusters_on_axial_slices(
    volume: np.ndarray,
    subgroup_masks: dict,
    quartiles: tuple[float, float, float] = (0.25, 0.5, 0.75),
    title: str = "Axial Subgroup Clusters",
    figsize: tuple = (18, 5),
    alpha: float = 0.4
) -> None:
    """
    Display axial slices at the 1st quartile, mid, and 3rd quartile
    with different subgroup clusters colored on the image.
    """
    z_dim = volume.shape[0]
    z_indices = [max(0, min(z_dim - 1, int(round(q * (z_dim - 1)))))
                 for q in quartiles]

    label_volume = np.zeros(volume.shape, dtype=np.uint8)
    label_order = ['E1', 'E2', 'E3', 'E4']
    for idx, name in enumerate(label_order, start=1):
        label_volume[subgroup_masks.get(name, np.zeros_like(volume, dtype=bool))] = idx

    cmap = ListedColormap(['black', '#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    legend_handles = [
        Patch(facecolor='#d62728', edgecolor='w', label='E1'),
        Patch(facecolor='#ff7f0e', edgecolor='w', label='E2'),
        Patch(facecolor='#2ca02c', edgecolor='w', label='E3'),
        Patch(facecolor='#1f77b4', edgecolor='w', label='E4'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    vmin = np.percentile(volume, 2)
    vmax = np.percentile(volume, 98)

    for ax, z in zip(axes, z_indices):
        img = volume[z, :, :]
        mask_slice = label_volume[z, :, :]
        ax.imshow(img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.imshow(mask_slice, cmap=cmap, alpha=alpha, origin='lower', vmin=0, vmax=4)
        ax.set_title(f"Axial Z={z}")
        ax.axis('off')

    axes[-1].legend(handles=legend_handles, loc='lower right', framealpha=0.85)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Optional: Collapsibility (Eq. 3)
# ─────────────────────────────────────────────
def compute_collapsibility(result_ins: EmphysemaResult,
                            result_exp: EmphysemaResult) -> dict:
    """
    Compute collapsibility of each subgroup between inspiratory
    and expiratory CT (Equation 3 in paper).

    C_i(%) = [1 - E_i_exp / E_i_ins] * 100
    """
    def safe_collapse(ins_vol, exp_vol):
        if ins_vol == 0:
            return float('nan')
        return (1 - exp_vol / ins_vol) * 100.0

    return {
        'C_E1': safe_collapse(result_ins.e1_volume_ml, result_exp.e1_volume_ml),
        'C_E2': safe_collapse(result_ins.e2_volume_ml, result_exp.e2_volume_ml),
        'C_E3': safe_collapse(result_ins.e3_volume_ml, result_exp.e3_volume_ml),
        'C_E4': safe_collapse(result_ins.e4_volume_ml, result_exp.e4_volume_ml),
    }


# ─────────────────────────────────────────────
# Main Pipeline Entry Point
# ─────────────────────────────────────────────
def run_pipeline(
    dicom_dir: str,
    dicom_dir_exp: Optional[str] = None   # optional expiratory CT
) -> EmphysemaResult:
    """
    Full pipeline for emphysema size classification from DICOM CT.

    Parameters
    ----------
    dicom_dir     : path to inspiratory CT DICOM directory
    dicom_dir_exp : path to expiratory CT DICOM directory (optional,
                    enables collapsibility computation)

    Returns
    -------
    EmphysemaResult for inspiratory CT (+ prints collapsibility if exp given)
    """
    print("\n=== INSPIRATORY CT ===")
    volume, spacing = load_dicom_series(dicom_dir)
    lung_mask = segment_lung(volume, spacing)
    emph_mask = extract_emphysema_mask(volume, lung_mask)
    cleaned_mask = noise_reduction(emph_mask)
    subgroups = size_based_emphysema_clustering(volume, cleaned_mask, emph_mask, spacing)
    visualize_subgroup_clusters_on_axial_slices(
        volume,
        subgroups,
        title="Axial Subgroup Clusters at 1st Quartile, Mid, 3rd Quartile"
    )

    result_ins = compute_emphysema_indices(subgroups, lung_mask, emph_mask, spacing)
    print("\n" + result_ins.summary())

    # if dicom_dir_exp:
    #     print("\n=== EXPIRATORY CT ===")
    #     vol_exp, spacing_exp = load_dicom_series(dicom_dir_exp)
    #     lung_exp = segment_lung(vol_exp, spacing_exp)
    #     emph_exp = extract_emphysema_mask(vol_exp, lung_exp)
    #     cleaned_exp = noise_reduction(emph_exp)
    #     subgroups_exp = size_based_emphysema_clustering(vol_exp, cleaned_exp, emph_exp, spacing_exp)
    #     result_exp = compute_emphysema_indices(subgroups_exp, lung_exp, emph_exp, spacing_exp)
    #     print("\n" + result_exp.summary())

    #     collapse = compute_collapsibility(result_ins, result_exp)
    #     print("\n=== Collapsibility (Eq. 3) ===")
    #     for k, v in collapse.items():
    #         print(f"  {k}: {v:.1f}%")

    return result_ins


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Single inspiratory CT
    result = run_pipeline(
        dicom_dir="D:\\emphysema\\data\\2246\\2020-01-29\\4",
        # dicom_dir_exp="/path/to/expiratory_dicom/"  # uncomment for paired CT
    )

    # Access individual subgroup masks for further analysis
    e3_mask = result.masks['E3']   # acinar-sized emphysema holes
