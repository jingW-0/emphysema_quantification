"""Tests for emphy_size_distance_iterative.py (Iterative EDT thresholding method)."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emphy_size_distance_iterative import (
    compute_edt,
    edt_iterative_clustering,
    compute_emphysema_indices,
)


class TestEDTComputation:
    def test_edt_shape_and_nonnegative(self, synthetic_emph_mask, synthetic_spacing):
        edt = compute_edt(synthetic_emph_mask, synthetic_spacing)
        assert edt.shape == synthetic_emph_mask.shape
        assert (edt >= 0).all()


class TestEDTClustering:
    def test_edt_iterative_basic(self, synthetic_emph_mask, synthetic_spacing):
        edt = compute_edt(synthetic_emph_mask, synthetic_spacing)
        cluster_map = edt_iterative_clustering(synthetic_emph_mask, edt, synthetic_spacing)
        # cluster_map should have same shape and integer labels
        assert cluster_map.shape == synthetic_emph_mask.shape
        assert issubclass(cluster_map.dtype.type, np.integer)
        # All classified voxels should be subset of emphysema mask
        assert np.all((cluster_map != 0) <= synthetic_emph_mask)


class TestIndicesCalculation:
    def test_compute_indices_returns_result(self, synthetic_emph_mask, synthetic_lung_mask, synthetic_spacing):
        edt = compute_edt(synthetic_emph_mask, synthetic_spacing)
        cluster_map = edt_iterative_clustering(synthetic_emph_mask, edt, synthetic_spacing)
        result = compute_emphysema_indices(cluster_map, synthetic_lung_mask, synthetic_emph_mask, synthetic_spacing)
        assert hasattr(result, 'laa_percent')
        assert hasattr(result, 'lung_volume_ml')
        assert isinstance(result.laa_percent, float)