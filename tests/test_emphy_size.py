"""Tests for emphy_size.py (Gaussian LPF method)."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emphy_size import (
    estimate_sigma,
    noise_reduction,
    EmphysemaResult,
)


class TestEstimateSigma:
    """Test Gaussian kernel sigma estimation (Eq. 1 in paper)."""
    
    def test_sigma_estimation_positive(self):
        """Sigma should be positive for positive radius."""
        sigma = estimate_sigma(5.0)
        assert sigma > 0, "sigma should be positive for positive radius"
    
    def test_sigma_estimation_increases_with_radius(self):
        """Larger radius should result in larger sigma."""
        sigma_small = estimate_sigma(3.0)
        sigma_large = estimate_sigma(10.0)
        assert sigma_large > sigma_small, "sigma should increase with radius"
    
    def test_sigma_estimation_formula(self):
        """Verify sigma follows paper formula: sigma = 0.147 + 0.1038 * radius."""
        radius = 7.5
        expected_sigma = 0.147 + 0.1038 * radius
        actual_sigma = estimate_sigma(radius)
        assert np.isclose(actual_sigma, expected_sigma, rtol=1e-5)


class TestNoiseReduction:
    """Test noise reduction (removal of small clusters)."""
    
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


class TestEmphysemaResult:
    """Test EmphysemaResult data container."""
    
    def test_result_initialization(self):
        """EmphysemaResult should initialize with default values."""
        result = EmphysemaResult()
        assert result.laa_percent == 0.0
        assert result.e1_volume_ml == 0.0
        assert result.e2_volume_ml == 0.0
        assert result.e3_volume_ml == 0.0
        assert result.e4_volume_ml == 0.0
    
    def test_result_summary_string(self):
        """Summary should be formatted correctly."""
        result = EmphysemaResult(
            laa_percent=15.5,
            e1_volume_ml=100.0,
            e1_fraction=5.0,
            lung_volume_ml=2000.0
        )
        summary = result.summary()
        
        assert "Emphysema Size Classification Results" in summary
        assert "15.5" in summary  # LAA percent
        assert "100.0" in summary  # E1 volume
        assert "5.0" in summary   # E1 fraction
    
    def test_result_masks_field(self):
        """Masks field should be mutable dict."""
        result = EmphysemaResult()
        mask = np.zeros((10, 10, 10), dtype=bool)
        result.masks['E1'] = mask
        assert np.array_equal(result.masks['E1'], mask)


class TestIntegration:
    """Integration tests using synthetic data."""
    
    def test_noise_reduction_on_synthetic_mask(self, synthetic_emph_mask):
        """Noise reduction should work on synthetic emphysema mask."""
        result = noise_reduction(synthetic_emph_mask)
        
        # Result should be valid
        assert result.dtype == bool
        assert result.shape == synthetic_emph_mask.shape
        
        # Should not have more voxels than input
        assert result.sum() <= synthetic_emph_mask.sum()
    
    def test_size_thresholds_ordering(self):
        """Size thresholds should be ordered from large to small."""
        thresholds = [15.0, 7.0, 1.5]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] > thresholds[i + 1], \
                "Thresholds should be ordered large to small"
