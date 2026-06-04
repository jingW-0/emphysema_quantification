"""
Emphysema Size Classification via Distance Transform + Watershed
================================================================
Proposed alternative to Oh et al. (2017)'s Gaussian LPF method.

Core idea:
  1. Compute Euclidean Distance Transform (EDT) on emphysema mask
     → each voxel's value = distance to nearest non-emphysema boundary (mm)
     → local maxima of EDT = centers of emphysema holes
     → EDT value at a local maximum = effective radius of that hole

  2. Use local maxima as seeds for marker-controlled watershed
     → watershed grows regions competitively from all seeds simultaneously
     → touching holes are separated at the natural ridge between their
        distance fields (analogous to cell nucleus segmentation)

  3. For each watershed region, the EDT value at its seed = hole radius
     → classify directly into E1/E2/E3/E4 by radius

Advantages over Gaussian LPF (Oh et al.):
  - No sigma estimation parameters (β₀, β₁)
  - Handles non-spherical holes naturally
  - Touching holes separated by watershed ridge (vs. underestimated by Gaussian blur)
  - Single distance transform pass vs. iterative filtering
  - Hole size is directly readable from EDT value — no indirect mapping

Dependencies:
    pip install SimpleITK scipy numpy scikit-image
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    distance_transform_edt,
    label,
    generate_binary_structure,
    maximum_filter,
    maximum,
    center_of_mass
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from dataclasses import dataclass, field
from typing import Optional
from time import perf_counter
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
LAA_THRESHOLD_HU  = -950
LUNG_THRESHOLD_HU = -400
NOISE_VOXEL_MIN   = 2

# Size thresholds in mm — same anatomical basis as Oh et al.
# Radius thresholds (half the diameter thresholds in paper)
# Paper uses diameter: <1.5, <7, <15, >=15 mm
# → radius: <0.75, <3.5, <7.5, >=7.5 mm
RADIUS_THRESHOLDS_MM = {
    'E1': (0.0,  0.75),   # alveolus / noise
    'E2': (0.75, 3.5),    # subacinus
    'E3': (3.5,  7.5),    # acinus / sublobular
    'E4': (7.5,  np.inf)  # extra-lobular
}

# H-maxima suppression threshold (mm)
# A peak must rise at least H mm above its surroundings to be a valid seed.
# h=1.0mm: anatomically grounded — smaller than E1 radius (0.75mm diameter
# boundary) so genuine holes are preserved, but shallow boundary artifacts
# from irregular emphysema surfaces are suppressed.
# Preserves EDT values intact (unlike Gaussian smoothing which pulls peaks down).
H_MAXIMA_MM = 1.0


# ─────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────
@dataclass
class EmphysemaResult:
    laa_percent: float = 0.0

    e1_volume_ml: float = 0.0
    e2_volume_ml: float = 0.0
    e3_volume_ml: float = 0.0
    e4_volume_ml: float = 0.0

    e1_fraction: float = 0.0
    e2_fraction: float = 0.0
    e3_fraction: float = 0.0
    e4_fraction: float = 0.0

    lung_volume_ml: float = 0.0
    voxel_size_ml:  float = 0.0

    masks: dict = field(default_factory=dict)
    # Per-hole data: list of (centroid_zyx, radius_mm, subgroup_label)
    hole_catalogue: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Emphysema Size Classification (Distance+Watershed) ===",
            f"  Lung volume       : {self.lung_volume_ml:.1f} mL",
            f"  Total %LAA        : {self.laa_percent:.2f}%",
            f"  E1 (<1.5mm diam)  : {self.e1_fraction:.2f}%  ({self.e1_volume_ml:.1f} mL)",
            f"  E2 (1.5-7mm)      : {self.e2_fraction:.2f}%  ({self.e2_volume_ml:.1f} mL)",
            f"  E3 (7-15mm)       : {self.e3_fraction:.2f}%  ({self.e3_volume_ml:.1f} mL)",
            f"  E4 (>=15mm)       : {self.e4_fraction:.2f}%  ({self.e4_volume_ml:.1f} mL)",
            f"  Total holes found : {len(self.hole_catalogue)}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────
def _format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"


def visualize_orthogonal_views(volume: np.ndarray,
                               mask: np.ndarray,
                               title: str = "Orthogonal Views",
                               figsize: tuple = (15, 5)) -> None:
    """
    Display mid-axial, mid-sagittal, and mid-coronal views of the volume and mask.

    Parameters
    ----------
    volume : 3D array (Z, Y, X)
        CT/image data
    mask : 3D bool array (Z, Y, X)
        Binary mask to overlay
    title : str
        Figure title
    figsize : tuple
        Figure size (width, height)
    """
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Create transparent red overlay colormap for mask
    colors = ['black', 'red']
    n_bins = 100
    cmap_mask = ListedColormap(colors)

    # AXIAL (Z slice) — view looking down from superior
    ax_axial = axes[0]
    ax_axial.imshow(volume[z_mid, :, :], cmap='gray', origin='lower')
    ax_axial.imshow(mask[z_mid, :, :], cmap=cmap_mask, alpha=0.3, origin='lower')
    ax_axial.set_title(f"Axial (Z={z_mid})")
    ax_axial.set_xlabel("X (left-right)")
    ax_axial.set_ylabel("Y (ant-post)")

    # SAGITTAL (Y slice) — view from right side
    ax_sag = axes[1]
    sag_img = volume[:, y_mid, :]
    sag_msk = mask[:, y_mid, :]
    ax_sag.imshow(sag_img, cmap='gray', origin='lower', aspect='auto')
    ax_sag.imshow(sag_msk, cmap=cmap_mask, alpha=0.3, origin='lower', aspect='auto')
    ax_sag.set_title(f"Sagittal (Y={y_mid})")
    ax_sag.set_xlabel("X (left-right)")
    ax_sag.set_ylabel("Z (sup-inf)")

    # CORONAL (X slice) — view from front
    ax_cor = axes[2]
    cor_img = volume[:, :, x_mid]
    cor_msk = mask[:, :, x_mid]
    ax_cor.imshow(cor_img, cmap='gray', origin='lower', aspect='auto')
    ax_cor.imshow(cor_msk, cmap=cmap_mask, alpha=0.3, origin='lower', aspect='auto')
    ax_cor.set_title(f"Coronal (X={x_mid})")
    ax_cor.set_xlabel("Y (ant-post)")
    ax_cor.set_ylabel("Z (sup-inf)")

    plt.tight_layout()
    plt.show()
    print(f"[visualize] Displayed orthogonal views: {title}")


def visualize_subgroup_masks_on_image(volume: np.ndarray,
                                      subgroup_masks: dict,
                                      title: str = "Subgroup Masks on Image",
                                      figsize: tuple = (15, 5)) -> None:
    """
    Overlay subgroup masks on axial slices at the 1st quartile, mid slice,
    and 3rd quartile using the same E1-E4 color mapping as emphy_size.py.
    """
    z_dim = volume.shape[0]
    z_indices = [
        int(round(0.25 * (z_dim - 1))),
        int(round(0.5 * (z_dim - 1))),
        int(round(0.75 * (z_dim - 1)))
    ]

    label_volume = np.zeros(volume.shape, dtype=np.uint8)
    label_order = ['E1', 'E2', 'E3', 'E4']
    for idx, name in enumerate(label_order, start=1):
        label_volume[subgroup_masks.get(name, np.zeros_like(volume, dtype=bool))] = idx

    cmap = ListedColormap(['black', '#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    vmin = np.nanpercentile(volume, 2)
    vmax = np.nanpercentile(volume, 98)

    for ax, z in zip(axes, z_indices):
        img = volume[z, :, :]
        mask_slice = label_volume[z, :, :]
        ax.imshow(img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.imshow(mask_slice, cmap=cmap, alpha=0.35, origin='lower', vmin=0, vmax=len(label_order))
        ax.set_title(f"Axial (Z={z})")
        ax.axis('off')

    legend_handles = [
        Patch(facecolor='#d62728', edgecolor='w', label='E1'),
        Patch(facecolor='#ff7f0e', edgecolor='w', label='E2'),
        Patch(facecolor='#2ca02c', edgecolor='w', label='E3'),
        Patch(facecolor='#1f77b4', edgecolor='w', label='E4'),
    ]
    axes[-1].legend(handles=legend_handles, loc='lower right', framealpha=0.8)

    plt.tight_layout()
    plt.show()
    print(f"[visualize] Displayed subgroup overlay: {title}")


# ─────────────────────────────────────────────
# Step 1: DICOM Loading
# ─────────────────────────────────────────────
def load_dicom_series(dicom_dir: str) -> tuple[np.ndarray, tuple]:
    """Load DICOM series. Returns (volume_HU, spacing_zyx_mm)."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM files in: {dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = image.GetSpacing()
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    print(f"[load] shape={volume.shape}, spacing(z,y,x)={spacing_zyx}")
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
# Step 3: Emphysema Mask + Noise Reduction
# ─────────────────────────────────────────────
def extract_emphysema_mask(volume: np.ndarray,
                            lung_mask: np.ndarray) -> np.ndarray:
    return (volume < LAA_THRESHOLD_HU) & lung_mask


def noise_reduction(emph_mask: np.ndarray,
                    min_voxels: int = NOISE_VOXEL_MIN) -> np.ndarray:
    """Remove connected components smaller than min_voxels.

    Operates in 3D on the full volume using 6-connectivity
    (equivalent to `generate_binary_structure(3, 1)`).
    `min_voxels` thresholds components by 3D voxel count.
    """
    struct = generate_binary_structure(3, 1)
    labeled, _ = label(emph_mask, structure=struct)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    cleaned = sizes[labeled] >= min_voxels
    print(f"[noise_reduction] {cleaned.sum():,} voxels remain")
    return cleaned


# ─────────────────────────────────────────────
# Step 4: Euclidean Distance Transform
# ─────────────────────────────────────────────
def compute_distance_transform(emph_mask: np.ndarray,
                                spacing_zyx: tuple) -> np.ndarray:
    """
    Compute EDT on the emphysema mask.

    EDT value at each voxel = distance in mm to the nearest
    non-emphysema boundary voxel.

    At a local maximum inside a hole, this value approximates
    the inscribed sphere radius — i.e., the effective hole radius.

    Parameters
    ----------
    emph_mask   : bool array, True = emphysema
    spacing_zyx : voxel spacing in mm

    Returns
    -------
    edt : float32 array, same shape, values in mm
    """
    # distance_transform_edt sampling= voxel sizes in each axis
    edt = distance_transform_edt(emph_mask, sampling=spacing_zyx)
    print(f"[EDT] max distance: {edt.max():.2f} mm  "
          f"(largest inscribed sphere radius)")
    return edt.astype(np.float32)


def visualize_distance_field(edt: np.ndarray,
                             mask: Optional[np.ndarray] = None,
                             title: str = "Distance Field",
                             figsize: tuple = (15, 5),
                             vmax_pct: float = 99.5) -> None:
    """
    Display mid-axial, mid-sagittal, and mid-coronal slices of the
    distance transform (in mm). Optionally overlay a binary mask.
    """
    z_mid = edt.shape[0] // 2
    y_mid = edt.shape[1] // 2
    x_mid = edt.shape[2] // 2

    # Robust color scaling: clip at percentile to avoid outlier domination
    vmax = float(np.nanpercentile(edt, vmax_pct))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    im0 = axes[0].imshow(edt[z_mid, :, :], cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    if mask is not None:
        axes[0].contour(mask[z_mid, :, :], colors='r', linewidths=0.5)
    axes[0].set_title(f"Axial EDT (Z={z_mid})")

    im1 = axes[1].imshow(edt[:, y_mid, :], cmap='viridis', origin='lower', aspect='auto', vmin=0, vmax=vmax)
    if mask is not None:
        axes[1].contour(mask[:, y_mid, :], colors='r', linewidths=0.5)
    axes[1].set_title(f"Sagittal EDT (Y={y_mid})")

    im2 = axes[2].imshow(edt[:, :, x_mid], cmap='viridis', origin='lower', aspect='auto', vmin=0, vmax=vmax)
    if mask is not None:
        axes[2].contour(mask[:, :, x_mid], colors='r', linewidths=0.5)
    axes[2].set_title(f"Coronal EDT (X={x_mid})")

    # colorbar across all subplots
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Distance (mm)')

    plt.tight_layout()
    plt.show()
    print(f"[visualize] Displayed distance field: {title}")


def visualize_seeds_on_edt(edt: np.ndarray,
                          seeds: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          title: str = "Seeds on EDT",
                          figsize: tuple = (15, 5)) -> None:
    """
    Overlay seed locations on mid-slices of the EDT.

    seeds : bool array same shape as edt indicating seed voxels.
    """
    if seeds.sum() == 0:
        print("[visualize] No seeds to display.")
        return

    z_mid = edt.shape[0] // 2
    y_mid = edt.shape[1] // 2
    x_mid = edt.shape[2] // 2

    # get seed coordinates
    coords = np.argwhere(seeds)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Axial
    axes[0].imshow(edt[z_mid, :, :], cmap='viridis', origin='lower')
    if mask is not None:
        axes[0].contour(mask[z_mid, :, :], colors='w', linewidths=0.4)
    ax_coords = coords[coords[:, 0] == z_mid]
    if ax_coords.size:
        ys = ax_coords[:, 1]
        xs = ax_coords[:, 2]
        axes[0].scatter(xs, ys, c='red', s=20, marker='x')
    axes[0].set_title(f"Axial (Z={z_mid})")

    # Sagittal (slice at y_mid): display (Z, X)
    axes[1].imshow(edt[:, y_mid, :], cmap='viridis', origin='lower', aspect='auto')
    if mask is not None:
        axes[1].contour(mask[:, y_mid, :], colors='w', linewidths=0.4)
    sag_coords = coords[coords[:, 1] == y_mid]
    if sag_coords.size:
        zs = sag_coords[:, 0]
        xs = sag_coords[:, 2]
        axes[1].scatter(xs, zs, c='red', s=20, marker='x')
    axes[1].set_title(f"Sagittal (Y={y_mid})")

    # Coronal (slice at x_mid): display (Z, Y)
    axes[2].imshow(edt[:, :, x_mid], cmap='viridis', origin='lower', aspect='auto')
    if mask is not None:
        axes[2].contour(mask[:, :, x_mid], colors='w', linewidths=0.4)
    cor_coords = coords[coords[:, 2] == x_mid]
    if cor_coords.size:
        zs = cor_coords[:, 0]
        ys = cor_coords[:, 1]
        axes[2].scatter(ys, zs, c='red', s=20, marker='x')
    axes[2].set_title(f"Coronal (X={x_mid})")

    plt.tight_layout()
    plt.show()
    print(f"[visualize] Displayed seeds overlay: {title}")


# ─────────────────────────────────────────────
# Step 5: Seed Detection via H-Maxima Suppression
# ─────────────────────────────────────────────
def h_maxima(edt: np.ndarray, h: float) -> np.ndarray:
    """
    H-maxima transform: suppress all maxima that do not rise
    at least h above their surrounding regional maximum.

    Definition:
        H-maxima(f, h) = f - R_f^delta(f - h)
        where R_f^delta is the geodesic reconstruction by dilation
        of (f - h) under f.

    In practice: a peak at height p is kept only if there is no
    path to a higher peak without descending more than h.
    Peaks separated by a saddle shallower than h are merged → one seed.

    This directly implements the physical criterion:
        "Two maxima represent distinct holes only if they are separated
         by a valley deeper than h mm in the distance field."

    h = 1.0mm means: if two candidate centers are connected without
    the distance field dropping more than 1mm between them, they are
    part of the same hole.

    Parameters
    ----------
    edt : distance transform array (values in mm)
    h   : suppression depth in mm (same units as edt)

    Returns
    -------
    hmax : array same shape as edt, with shallow maxima suppressed
           True where regional maxima survive suppression
    """
    from skimage.morphology import reconstruction

    # Geodesic reconstruction by dilation of (edt - h) under edt
    seed = np.clip(edt - h, 0, None)          # shift surface down by h
    reconstructed = reconstruction(seed, edt, method='dilation')

    # Residue: voxels where edt rises more than h above reconstructed
    residue = edt - reconstructed              # > 0 only at significant peaks

    # Regional maxima of residue = surviving h-maxima
    # A voxel is a regional maximum if it is strictly greater than
    # all its neighbors in the reconstructed image
    struct = generate_binary_structure(3, 1)   # 6-connectivity
    local_max = maximum_filter(residue, footprint=struct) == residue
    hmax_mask = local_max & (residue > 0) & (edt > 0)

    return hmax_mask


def find_seeds(edt: np.ndarray,
               emph_mask: np.ndarray,
               h_mm: float = H_MAXIMA_MM) -> np.ndarray:
    """
    Find watershed seeds using h-maxima suppression on the EDT.

    Design rationale:
      - H-maxima preserves EDT values intact at surviving peaks
        → radius estimates remain geometrically accurate
      - h=1.0mm has clear anatomical meaning: two candidate centers
        must be separated by a valley >1mm deep to count as distinct holes
      - No Gaussian smoothing needed: smoothing would pull peak values
        downward, systematically underestimating small hole radii

    Parameters
    ----------
    edt       : distance transform in mm
    emph_mask : emphysema mask (seeds restricted to emphysema only)
    h_mm      : suppression threshold in mm

    Returns
    -------
    seed_mask : bool array, True at seed locations
    """
    hmax_mask = h_maxima(edt, h=h_mm)

    # Restrict seeds to emphysema mask
    seed_mask = hmax_mask & emph_mask

    print(f"[seeds] {seed_mask.sum():,} seeds "
          f"(h-maxima suppression, h={h_mm}mm)")
    return seed_mask


# ─────────────────────────────────────────────
# Step 6: Marker-Controlled Watershed
# ─────────────────────────────────────────────
def watershed_segmentation(edt: np.ndarray,
                            seed_mask: np.ndarray,
                            emph_mask: np.ndarray) -> np.ndarray:
    """
    Watershed segmentation of emphysema holes.

    Watershed "floods" from seed points into the distance field.
    Regions grow competitively — two adjacent holes are naturally
    separated at the ridge between their distance fields, which
    corresponds to their geometric boundary.

    Uses -edt as the "topographic surface" so that:
      - Seeds are at valleys (high EDT = high distance = center of hole)
      - Watershed fills upward (toward lower EDT = toward boundary)

    Parameters
    ----------
    edt       : distance transform in mm
    seed_mask : bool mask of seed locations
    emph_mask : bool mask constraining watershed to emphysema only

    Returns
    -------
    labels : int array, each unique label = one emphysema hole instance
    """
    # Label each seed uniquely
    seed_labels, n_seeds = label(seed_mask)

    # Watershed on inverted EDT (high distance = low "elevation" = seed)
    ws_labels = watershed(
        -edt,
        markers=seed_labels,
        mask=emph_mask,
        compactness=0.01   # small compactness → follow distance ridges closely
    )

    print(f"[watershed] {n_seeds} holes segmented")
    return ws_labels


# ─────────────────────────────────────────────
# Step 7: Classify Holes by Radius
# ─────────────────────────────────────────────
def classify_holes(ws_labels: np.ndarray,
                   edt: np.ndarray,
                   spacing_zyx: tuple) -> tuple[dict, list]:
    """
    Classify each watershed region by the EDT value at its seed.

    The EDT value at the local maximum of a region = inscribed sphere
    radius in mm → directly gives hole size without any parameter tuning.

    Returns
    -------
    subgroup_masks : dict {'E1','E2','E3','E4'} → bool arrays
    hole_catalogue : list of dicts, one per hole, with:
                     label, centroid_zyx, radius_mm, subgroup, volume_ml
    """
    voxel_vol_ml = np.prod(spacing_zyx) / 1000.0
    unique_labels = np.unique(ws_labels)
    unique_labels = unique_labels[unique_labels > 0]

    subgroup_masks = {k: np.zeros(ws_labels.shape, dtype=bool)
                      for k in ['E1', 'E2', 'E3', 'E4']}
    hole_catalogue = []

    if unique_labels.size == 0:
        return subgroup_masks, hole_catalogue

    max_radii = maximum(edt, labels=ws_labels, index=unique_labels)
    centroids = center_of_mass(np.ones(ws_labels.shape, dtype=np.int32),
                               labels=ws_labels,
                               index=unique_labels)
    label_counts = np.bincount(ws_labels.ravel())

    group_names = ['E1', 'E2', 'E3', 'E4']
    group_to_index = {name: idx for idx, name in enumerate(group_names)}
    max_label = int(unique_labels.max())
    label_to_group = np.full(max_label + 1, -1, dtype=np.int8)

    for lbl, radius_mm, centroid in zip(unique_labels,
                                       max_radii,
                                       centroids):
        radius_mm = float(radius_mm)
        subgroup = _assign_subgroup(radius_mm)
        label_to_group[int(lbl)] = group_to_index[subgroup]

        hole_catalogue.append({
            'label':        int(lbl),
            'centroid_zyx': [float(c) for c in centroid],
            'radius_mm':    radius_mm,
            'diameter_mm':  radius_mm * 2,
            'subgroup':     subgroup,
            'volume_ml':    float(label_counts[int(lbl)]) * voxel_vol_ml
        })

    group_labels = label_to_group[ws_labels]
    for name, idx in group_to_index.items():
        subgroup_masks[name] = group_labels == idx

    hole_catalogue.sort(key=lambda x: x['radius_mm'], reverse=True)

    for k, v in subgroup_masks.items():
        print(f"[classify] {k}: {v.sum():,} voxels")

    return subgroup_masks, hole_catalogue

# def classify_holes(ws_labels: np.ndarray,
#                    edt: np.ndarray,
#                    spacing_zyx: tuple) -> tuple[dict, list]:
#     """
#     Classify each watershed region by the EDT value at its seed.
 
#     The EDT value at the local maximum of a region = inscribed sphere
#     radius in mm → directly gives hole size without any parameter tuning.
 
#     Returns
#     -------
#     subgroup_masks : dict {'E1','E2','E3','E4'} → bool arrays
#     hole_catalogue : list of dicts, one per hole, with:
#                      label, centroid_zyx, radius_mm, subgroup, volume_ml
#     """
#     voxel_vol_ml = np.prod(spacing_zyx) / 1000.0
#     unique_labels = np.unique(ws_labels)
#     unique_labels = unique_labels[unique_labels > 0]
 
#     subgroup_masks = {k: np.zeros(ws_labels.shape, dtype=bool)
#                       for k in ['E1', 'E2', 'E3', 'E4']}
#     hole_catalogue = []
 
#     for lbl in unique_labels:
#         region = ws_labels == lbl
 
#         # Radius = EDT value at the peak voxel within this region
#         region_edt = edt * region
#         peak_idx = np.unravel_index(region_edt.argmax(), edt.shape)
#         radius_mm = float(edt[peak_idx])
 
#         # Classify by radius
#         subgroup = _assign_subgroup(radius_mm)
#         subgroup_masks[subgroup] |= region
 
#         # Centroid
#         coords = np.argwhere(region)
#         centroid = coords.mean(axis=0)
 
#         hole_catalogue.append({
#             'label':       int(lbl),
#             'centroid_zyx': centroid.tolist(),
#             'radius_mm':   radius_mm,
#             'diameter_mm': radius_mm * 2,
#             'subgroup':    subgroup,
#             'volume_ml':   region.sum() * voxel_vol_ml
#         })
 
#     # Sort by size descending
#     hole_catalogue.sort(key=lambda x: x['radius_mm'], reverse=True)
 
#     for k, v in subgroup_masks.items():
#         print(f"[classify] {k}: {v.sum():,} voxels")
 
#     return subgroup_masks, hole_catalogue

def _assign_subgroup(radius_mm: float) -> str:
    for name, (lo, hi) in RADIUS_THRESHOLDS_MM.items():
        if lo <= radius_mm < hi:
            return name
    return 'E4'  # fallback for very large


# ─────────────────────────────────────────────
# Step 8: Compute Indices
# ─────────────────────────────────────────────
def compute_emphysema_indices(subgroup_masks: dict,
                               hole_catalogue: list,
                               lung_mask: np.ndarray,
                               emph_mask: np.ndarray,
                               spacing_zyx: tuple) -> EmphysemaResult:
    voxel_vol_ml = np.prod(spacing_zyx) / 1000.0
    lung_vol_ml  = lung_mask.sum() * voxel_vol_ml
    laa_pct = 100.0 * emph_mask.sum() / max(lung_mask.sum(), 1)

    def frac(mask):
        return 100.0 * mask.sum() / max(lung_mask.sum(), 1)

    return EmphysemaResult(
        laa_percent   = laa_pct,
        e1_volume_ml  = subgroup_masks['E1'].sum() * voxel_vol_ml,
        e2_volume_ml  = subgroup_masks['E2'].sum() * voxel_vol_ml,
        e3_volume_ml  = subgroup_masks['E3'].sum() * voxel_vol_ml,
        e4_volume_ml  = subgroup_masks['E4'].sum() * voxel_vol_ml,
        e1_fraction   = frac(subgroup_masks['E1']),
        e2_fraction   = frac(subgroup_masks['E2']),
        e3_fraction   = frac(subgroup_masks['E3']),
        e4_fraction   = frac(subgroup_masks['E4']),
        lung_volume_ml = lung_vol_ml,
        voxel_size_ml  = voxel_vol_ml,
        masks          = subgroup_masks,
        hole_catalogue = hole_catalogue
    )


# ─────────────────────────────────────────────
# Optional: Collapsibility (Eq. 3, Oh et al.)
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
# Main Pipeline
# ─────────────────────────────────────────────
def run_pipeline(dicom_dir: str,
                 dicom_dir_exp: Optional[str] = None,
                 h_mm: float = H_MAXIMA_MM
                 ) -> EmphysemaResult:
    """
    Full distance transform + h-maxima + watershed emphysema size pipeline.

    Parameters
    ----------
    dicom_dir     : inspiratory CT DICOM directory
    dicom_dir_exp : expiratory CT DICOM directory (optional)
    h_mm          : h-maxima suppression depth in mm (default 1.0mm)
                    increase if too many spurious seeds
                    decrease if genuine small holes are being missed
    """
    print("\n=== INSPIRATORY CT ===")
    t0 = perf_counter()
    volume, spacing   = load_dicom_series(dicom_dir)
    t1 = perf_counter()
    lung_mask         = segment_lung(volume, spacing)
    t2 = perf_counter()
    visualize_orthogonal_views(volume, lung_mask, title="Inspiratory CT - Lung Mask")
    emph_mask         = extract_emphysema_mask(volume, lung_mask)
    visualize_orthogonal_views(volume, emph_mask, title="Inspiratory CT - Emphysema Mask")
    t3 = perf_counter()
    cleaned           = noise_reduction(emph_mask)
    visualize_orthogonal_views(volume, cleaned, title="Inspiratory CT - Cleaned Emphysema Mask")
    t4 = perf_counter()
    edt               = compute_distance_transform(cleaned, spacing)
    t5 = perf_counter()
    visualize_distance_field(edt, mask=cleaned, title="Inspiratory EDT - Distance Field")
    seeds             = find_seeds(edt, cleaned, h_mm)
    visualize_seeds_on_edt(edt, seeds, mask=cleaned, title="Seeds on Inspiratory EDT")
    t6 = perf_counter()
    ws_labels         = watershed_segmentation(edt, seeds, cleaned)
    t7 = perf_counter()
    subgroups, holes  = classify_holes(ws_labels, edt, spacing)
    visualize_subgroup_masks_on_image(volume, subgroups, title="Subgroup Masks on Inspiratory Image")
    t8 = perf_counter()
    result_ins        = compute_emphysema_indices(
                            subgroups, holes, lung_mask, emph_mask, spacing)
    t9 = perf_counter()

    print("\n=== TIMING ===")
    print(f"  load_dicom_series        : {_format_duration(t1 - t0)}")
    print(f"  segment_lung             : {_format_duration(t2 - t1)}")
    print(f"  extract_emphysema_mask   : {_format_duration(t3 - t2)}")
    print(f"  noise_reduction          : {_format_duration(t4 - t3)}")
    print(f"  compute_distance_transform: {_format_duration(t5 - t4)}")
    print(f"  find_seeds               : {_format_duration(t6 - t5)}")
    print(f"  watershed_segmentation   : {_format_duration(t7 - t6)}")
    print(f"  classify_holes           : {_format_duration(t8 - t7)}")
    print(f"  compute_emphysema_indices: {_format_duration(t9 - t8)}")
    print(f"  total inspiratory        : {_format_duration(t9 - t0)}")

    print("\n" + result_ins.summary())

    # if dicom_dir_exp:
    #     print("\n=== EXPIRATORY CT ===")
    #     t0_e = perf_counter()
    #     vol_e, sp_e      = load_dicom_series(dicom_dir_exp)
    #     t1_e = perf_counter()
    #     lung_e           = segment_lung(vol_e, sp_e)
    #     t2_e = perf_counter()
    #     visualize_orthogonal_views(vol_e, lung_e, title="Expiratory CT - Lung Mask")
    #     emph_e           = extract_emphysema_mask(vol_e, lung_e)
    #     t3_e = perf_counter()
    #     clean_e          = noise_reduction(emph_e)
    #     t4_e = perf_counter()
    #     edt_e            = compute_distance_transform(clean_e, sp_e)
    #     t5_e = perf_counter()
    #     seeds_e          = find_seeds(edt_e, clean_e, h_mm)
    #     t6_e = perf_counter()
    #     ws_e             = watershed_segmentation(edt_e, seeds_e, clean_e)
    #     t7_e = perf_counter()
    #     sub_e, holes_e   = classify_holes(ws_e, edt_e, sp_e)
    #     t8_e = perf_counter()
    #     result_exp       = compute_emphysema_indices(
    #                            sub_e, holes_e, lung_e, emph_e, sp_e)
    #     t9_e = perf_counter()

    #     print("\n=== TIMING ===")
    #     print(f"  load_dicom_series        : {_format_duration(t1_e - t0_e)}")
    #     print(f"  segment_lung             : {_format_duration(t2_e - t1_e)}")
    #     print(f"  extract_emphysema_mask   : {_format_duration(t3_e - t2_e)}")
    #     print(f"  noise_reduction          : {_format_duration(t4_e - t3_e)}")
    #     print(f"  compute_distance_transform: {_format_duration(t5_e - t4_e)}")
    #     print(f"  find_seeds               : {_format_duration(t6_e - t5_e)}")
    #     print(f"  watershed_segmentation   : {_format_duration(t7_e - t6_e)}")
    #     print(f"  classify_holes           : {_format_duration(t8_e - t7_e)}")
    #     print(f"  compute_emphysema_indices: {_format_duration(t9_e - t8_e)}")
    #     print(f"  total expiratory         : {_format_duration(t9_e - t0_e)}")

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
    result = run_pipeline(
        dicom_dir="D:\\emphysema\\data\\2246\\2020-01-29\\4",
        # dicom_dir_exp="/path/to/expiratory_dicom/",
        h_mm=1.0    # increase if over-seeding; decrease if missing small holes
    )

    # Inspect individual holes — unique to this method vs. Oh et al.
    large_holes = [h for h in result.hole_catalogue if h['subgroup'] == 'E4']
    print(f"\nLarge holes (E4, >=15mm diameter): {len(large_holes)}")
    for h in large_holes[:5]:
        print(f"  radius={h['radius_mm']:.1f}mm  vol={h['volume_ml']:.2f}mL  "
              f"centroid={[round(c,1) for c in h['centroid_zyx']]}")