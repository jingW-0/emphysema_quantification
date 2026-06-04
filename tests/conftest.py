"""Pytest fixtures and test utilities for emphysema pipeline."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def synthetic_ct_volume():
    """
    Create a synthetic CT volume for testing.
    
    Returns a 3D array with:
    - Background: -1000 HU (air)
    - Lung tissue: -500 HU (typical)
    - Emphysema pockets: -950 to -980 HU (LAA regions)
    """
    shape = (50, 128, 128)  # z, y, x
    volume = np.full(shape, -1000, dtype=np.float32)  # background air
    
    # Create lung region (central cube)
    lung_region = volume[10:40, 30:98, 30:98]
    lung_region[:] = -500  # lung tissue HU
    
    # Add emphysema pockets (spherical regions with LAA values)
    z_center, y_center, x_center = 20, 64, 64
    radius = 8
    for z in range(max(0, z_center - radius), min(shape[0], z_center + radius)):
        for y in range(max(0, y_center - radius), min(shape[1], y_center + radius)):
            for x in range(max(0, x_center - radius), min(shape[2], x_center + radius)):
                dist = np.sqrt((z - z_center)**2 + (y - y_center)**2 + (x - x_center)**2)
                if dist < radius:
                    volume[z, y, x] = -960  # emphysema HU
    
    return volume


@pytest.fixture
def synthetic_spacing():
    """Return typical voxel spacing in mm (z, y, x)."""
    return (2.0, 0.625, 0.625)  # High-resolution CT typical spacing


@pytest.fixture
def synthetic_lung_mask(synthetic_ct_volume):
    """Create a synthetic lung mask from the volume."""
    return (synthetic_ct_volume < -400).astype(bool)


@pytest.fixture
def synthetic_emph_mask(synthetic_ct_volume):
    """Create a synthetic emphysema mask from the volume."""
    return (synthetic_ct_volume <= -950).astype(bool)


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return tmp_path


# Constants for testing
TEST_SIZE_THRESHOLDS = [15.0, 7.0, 1.5]
TEST_RADIUS_THRESHOLDS = {
    'E1': (0.0,  0.75),
    'E2': (0.75, 3.5),
    'E3': (3.5,  7.5),
    'E4': (7.5,  np.inf)
}
