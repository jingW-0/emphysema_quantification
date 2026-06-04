"""Tests for emphy_size_distance.py (Distance Transform + Watershed method)."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emphy_size_distance import (
    noise_reduction,
    compute_distance_transform,
    EmphysemaResult,
)


class TestComputeDistanceTransform:
    """Test distance transform computation."""
    
    def test_distance_transform_shape(self, synthetic_emph_mask):
        """Distance transform should match input shape."""
        edt = compute_distance_transform(synthetic_emph_mask, (1.0, 1.0, 1.0))
        assert edt.shape == synthetic_emph_mask.shape
    
    def test_distance_transform_nonnegative(self, synthetic_emph_mask):
        """Distance transform values should be non-negative."""
        edt = compute_distance_transform(synthetic_emph_mask, (1.0, 1.0, 1.0))
        assert (edt >= 0).all()
    
    def test_distance_transform_zero_outside_mask(self, synthetic_emph_mask):
        """EDT should be zero outside emphysema mask."""
        edt = compute_distance_transform(synthetic_emph_mask, (1.0, 1.0, 1.0))
        background = ~synthetic_emph_mask
        assert (edt[background] == 0).all()


class TestNoiseReductionDistance:
    """Test noise reduction for distance transform method."""
    
    def test_noise_reduction_removes_small_clusters(self):
        """Clusters < 2 voxels should be removed."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        
        # Add a small isolated cluster (1 voxel)
        mask[5, 5, 5] = True
        
        # Add a larger cluster (5 voxels)
        mask[3:5, 3:5, 3] = True
        
        result = noise_reduction(mask)
        
        # Isolated voxel should be removed
        assert not result[5, 5, 5], "Isolated voxel should be removed"
        
        # Larger cluster should remain
        assert result[3:5, 3:5, 3].sum() > 0, "Larger cluster should remain"
    
    def test_noise_reduction_preserves_size(self):
        """Output should be same shape as input."""
        mask = np.random.rand(20, 30, 40) > 0.9
        result = noise_reduction(mask)
        assert result.shape == mask.shape
    
    def test_noise_reduction_output_type(self):
        """Output should be boolean array."""
        mask = np.random.rand(15, 15, 15) > 0.8
        result = noise_reduction(mask)
        assert result.dtype == bool


class TestComputeEmphysemaIndices:
    """Test emphysema index computation."""
    
    def test_indices_output_types(self):
        """All index fields should be numeric."""
        lung_mask = np.ones((10, 10, 10), dtype=bool)
        emph_mask = np.zeros((10, 10, 10), dtype=bool)
        emph_mask[4:6, 4:6, 4:6] = True
        
        subgroup_masks = {
            'E1': np.zeros_like(emph_mask),
            'E2': emph_mask.copy(),
            'E3': np.zeros_like(emph_mask),
            'E4': np.zeros_like(emph_mask),
        }
        
        hole_catalogue = [
            {'label': 0, 'radius_mm': 1.0, 'subgroup': 'E2', 'voxels': 8}
        ]
        
        spacing = (1.0, 1.0, 1.0)
        from emphy_size_distance import compute_emphysema_indices
        result = compute_emphysema_indices(
            subgroup_masks, hole_catalogue, lung_mask, emph_mask, spacing
        )
        
        assert isinstance(result, EmphysemaResult)
        assert isinstance(result.laa_percent, (float, np.floating))
        assert isinstance(result.e1_volume_ml, (float, np.floating))
        assert isinstance(result.lung_volume_ml, (float, np.floating))


class TestRadiusThresholds:
    """Test radius-based size classification."""
    
    def test_radius_thresholds_ordering(self):
        """Radius thresholds should be ordered consistently."""
        thresholds = {
            'E1': (0.0,  0.75),
            'E2': (0.75, 3.5),
            'E3': (3.5,  7.5),
            'E4': (7.5,  np.inf)
        }
        
        # Each lower bound should match previous upper bound (continuity)
        assert thresholds['E2'][0] == thresholds['E1'][1]
        assert thresholds['E3'][0] == thresholds['E2'][1]
        assert thresholds['E4'][0] == thresholds['E3'][1]
    
    def test_radius_classification_logic(self):
        """Verify radius-to-category classification."""
        thresholds = {
            'E1': (0.0,  0.75),
            'E2': (0.75, 3.5),
            'E3': (3.5,  7.5),
            'E4': (7.5,  np.inf)
        }
        
        # Test specific radii
        test_cases = [
            (0.5, 'E1'),
            (2.0, 'E2'),
            (5.0, 'E3'),
            (10.0, 'E4'),
        ]
        
        for radius, expected_category in test_cases:
            for category, (lower, upper) in thresholds.items():
                if lower <= radius < upper:
                    assert category == expected_category
                    break


class TestIntegration:
    """Integration tests using synthetic data."""
    
    def test_distance_pipeline_on_synthetic_data(self, synthetic_emph_mask, synthetic_spacing):
        """Test distance transform computation with synthetic data."""
        # Compute EDT in mm
        edt_mm = compute_distance_transform(synthetic_emph_mask, synthetic_spacing)
        
        # EDT should have same shape
        assert edt_mm.shape == synthetic_emph_mask.shape
        
        # EDT should be non-negative
        assert (edt_mm >= 0).all()
        
        # EDT should be zero outside emphysema mask
        background = ~synthetic_emph_mask
        assert (edt_mm[background] == 0).all()
    
    def test_noise_reduction_on_synthetic_mask(self, synthetic_emph_mask):
        """Noise reduction should work on synthetic emphysema mask."""
        result = noise_reduction(synthetic_emph_mask)
        
        # Result should be valid
        assert result.dtype == bool
        assert result.shape == synthetic_emph_mask.shape
        
        # Should not have more voxels than input
        assert result.sum() <= synthetic_emph_mask.sum()

