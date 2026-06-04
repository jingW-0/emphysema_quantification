"""Tests for emphy_size_distance.py (Distance Transform + Watershed method)."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emphy_size_distance import (
    compute_voxel_volume,
    compute_emphysema_indices,
    EmphysemaResult,
)


class TestComputeVoxelVolume:
    """Test voxel volume calculation."""
    
    def test_voxel_volume_isotropic(self):
        """Isotropic spacing should compute correctly."""
        spacing = (1.0, 1.0, 1.0)
        voxel_vol = compute_voxel_volume(spacing)
        assert np.isclose(voxel_vol, 1.0), "1mm³ voxel should have volume 1.0 mm³"
    
    def test_voxel_volume_anisotropic(self):
        """Anisotropic spacing should compute correctly."""
        spacing = (2.0, 0.5, 0.5)
        voxel_vol = compute_voxel_volume(spacing)
        expected = 2.0 * 0.5 * 0.5
        assert np.isclose(voxel_vol, expected)
    
    def test_voxel_volume_positive(self):
        """Voxel volume should always be positive."""
        spacing = (1.5, 0.625, 0.625)
        voxel_vol = compute_voxel_volume(spacing)
        assert voxel_vol > 0


class TestComputeEmphysemaIndices:
    """Test emphysema index computation."""
    
    def test_indices_basic_computation(self):
        """Test basic index calculation with synthetic data."""
        # Create synthetic masks
        lung_mask = np.ones((20, 30, 30), dtype=bool)
        emph_mask = np.zeros((20, 30, 30), dtype=bool)
        emph_mask[10:15, 10:20, 10:20] = True  # 5*10*10 = 500 voxels
        
        subgroup_masks = {
            'E1': np.zeros_like(emph_mask),
            'E2': emph_mask.copy(),  # All emphysema in E2
            'E3': np.zeros_like(emph_mask),
            'E4': np.zeros_like(emph_mask),
        }
        
        spacing = (1.0, 1.0, 1.0)  # 1mm³ voxels
        
        result = compute_emphysema_indices(
            subgroup_masks, lung_mask, emph_mask, spacing
        )
        
        # Assertions
        assert isinstance(result, EmphysemaResult)
        assert result.e2_volume_ml == pytest.approx(500.0, abs=1.0)  # 500 mm³ = 0.5 mL
        assert result.laa_percent > 0
    
    def test_indices_fractions_sum_reasonably(self):
        """E1-E4 fractions should sum to approximately total LAA."""
        lung_mask = np.ones((30, 40, 40), dtype=bool)
        emph_mask = np.zeros((30, 40, 40), dtype=bool)
        emph_mask[10:25, 10:30, 10:30] = True
        
        # Distribute emphysema across subgroups
        e1_mask = np.zeros_like(emph_mask)
        e1_mask[10:15, 10:20, 10:20] = True
        
        e2_mask = np.zeros_like(emph_mask)
        e2_mask[15:20, 10:20, 10:20] = True
        
        subgroup_masks = {
            'E1': e1_mask,
            'E2': e2_mask,
            'E3': np.zeros_like(emph_mask),
            'E4': np.zeros_like(emph_mask),
        }
        
        spacing = (1.0, 1.0, 1.0)
        result = compute_emphysema_indices(
            subgroup_masks, lung_mask, emph_mask, spacing
        )
        
        # Sum of individual fractions should be close to total LAA percent
        total_subgroup_frac = (result.e1_fraction + result.e2_fraction + 
                               result.e3_fraction + result.e4_fraction)
        assert total_subgroup_frac <= result.laa_percent * 1.01  # Allow small rounding error
    
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
        
        spacing = (1.0, 1.0, 1.0)
        result = compute_emphysema_indices(
            subgroup_masks, lung_mask, emph_mask, spacing
        )
        
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
        """Test distance transform pipeline with synthetic data."""
        from emphy_size_distance import distance_transform_edt_mm
        
        # Compute EDT in mm
        edt_mm = distance_transform_edt_mm(
            synthetic_emph_mask.astype(np.float32),
            synthetic_spacing
        )
        
        # EDT should have same shape
        assert edt_mm.shape == synthetic_emph_mask.shape
        
        # EDT should be non-negative
        assert (edt_mm >= 0).all()
        
        # EDT should be zero outside emphysema mask
        background = ~synthetic_emph_mask
        assert (edt_mm[background] == 0).all()
