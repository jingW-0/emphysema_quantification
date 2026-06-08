# Emphysema Quantification Pipeline

A Python framework for automated emphysema size classification from high-resolution CT scans, implementing two complementary algorithms for anatomical hole sizing.

## Background

Emphysema is characterized by destruction of alveolar walls, creating air-filled cavities (holes) of varying sizes. Clinical significance depends on distribution and size:

- **E1** (<1.5 mm): Alveolar-level damage
- **E2** (1.5–7 mm): Subacinar disease  
- **E3** (7–15 mm): Acinar/sublobular disease
- **E4** (≥15 mm): Paraseptal/extra-lobular emphysema

This codebase provides two independent methods for automated classification of emphysema holes.

## Algorithms

### 1. Size-Based Classification via Gaussian LPF (`emphy_size.py`)

Based on: **Oh et al. (2017)** — "Size variation and collapse of emphysema holes at inspiration and expiration CT scan: evaluation with modified length scale method and image co-registration" — *International Journal of COPD* 12:2043–2057

**Pipeline:**
1. Load DICOM series and extract HU values
2. Segment lung via thresholding + morphological operations
3. Extract emphysema mask (Low Attenuation Area, LAA ≤ -950 HU)
4. Noise reduction: remove connected components < 2 voxels
5. **Iterative Gaussian LPF:** Process from large to small kernel sizes
   - Apply Gaussian filter to current mask
   - Identify skeleton voxels (local maxima ≥ 99.9% of peak)
   - Dilate by anatomical radius
   - Intersect with original emphysema mask
   - Subtract classified region from current mask; repeat
6. Compute emphysema indices by subgroup size

**Key parameters:**
- Gaussian sigma estimation: sigma = 0.147 + 2 * 0.1038 * gamma, where gamma is radius in mm
- Diameter thresholds: [15.0, 7.0, 1.5] mm (converted to radii [7.5, 3.5, 0.75] mm)

**Pros:**
- Established in literature; validated methodology
- Smooth, continuous size spectrum via kernel scale

**Cons:**
- Requires sigma estimation parameters
- Iterative filtering may underestimate hole sizes at boundaries
- Gaussian blur can merge nearby holes

---

### 2. Size Classification via Distance Transform + Watershed (`emphy_size_distance.py`)

**Proposed alternative:** Direct hole segmentation and sizing via Euclidean distance.

**Pipeline:**
1. Load DICOM, segment lung, extract LAA mask (same as above)
2. Noise reduction
3. **Distance Transform:** Compute Euclidean distance (mm) from each emphysema voxel to the nearest non-emphysema boundary
   - Peak of EDT map = center of hole
   - EDT value at peak = hole radius
4. **Seed detection:** Local maxima of EDT with h-maxima suppression (H = 1.0 mm)
5. **Watershed segmentation:** Region-growing from all seeds simultaneously
   - Regions grow competitively; boundaries settle at ridges (zero crossings of EDT gradient)
   - Touching holes naturally separated
6. **Direct classification:** EDT value at each region's seed → radius → size class

**Key parameters:**
- EDT computed in physical mm (via voxel spacing)
- h-maxima threshold: 1.0 mm (suppresses shallow boundary artifacts)
- Radius thresholds: [0.75, 3.5, 7.5] mm → [E1, E2, E3, E4]

**Pros:**
- Physically interpretable: hole radius directly readable from EDT
- No parameter estimation required
- Single pass (non-iterative)
- Naturally handles non-spherical holes
- Touches holes separated by watershed ridge

**Cons:**
- Assumes holes are distinguishable in EDT (may fail in densely clustered emphysema)
- Watershed sensitive to seed selection

---

## Installation

### Requirements
- Python 3.8+
- Anaconda/Miniconda (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/jingW-0/emphysema_quantification.git
cd emphysema_quantification

# Create conda environment
conda create -n emphysema python=3.11
conda activate emphysema

# Install dependencies
pip install SimpleITK scipy numpy scikit-image matplotlib pydicom
```

## Testing

### Unit Tests

The project includes comprehensive unit tests for both algorithms.

**Install test dependencies:**
```bash
pip install -r requirements-dev.txt
```

**Run all tests:**
```bash
pytest
```

**Run tests with coverage report:**
```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html to view coverage
```

**Run specific test file:**
```bash
pytest tests/test_emphy_size.py -v
```

**Run specific test class or function:**
```bash
pytest tests/test_emphy_size.py::TestNoiseReduction -v
pytest tests/test_emphy_size.py::TestNoiseReduction::test_noise_reduction_removes_small_clusters -v
```

**Run tests in parallel (faster):**
```bash
pytest -n auto
```

### Code Quality

**Format code (auto-fix):**
```bash
black emphy_size.py emphy_size_distance.py tests/
isort emphy_size.py emphy_size_distance.py tests/
```

**Check code quality:**
```bash
flake8 emphy_size.py emphy_size_distance.py tests/
mypy emphy_size.py emphy_size_distance.py --ignore-missing-imports
```

**Run all checks with tox (multiple Python versions):**
```bash
tox                    # Run tests on all Python versions
tox -e coverage        # Generate coverage report
tox -e lint            # Run linting checks
tox -e format          # Auto-format code
tox -e typecheck       # Run type checking
```

### Continuous Integration

GitHub Actions automatically runs tests on every push and pull request:
- Tests run on Python 3.9, 3.10, 3.11
- Tests run on Linux, Windows, macOS
- Code coverage is uploaded to Codecov
- Linting checks enforce code quality

See [`.github/workflows/tests.yml`](.github/workflows/tests.yml) for CI configuration.

### Test Structure

```
tests/
├── __init__.py                 # Test package
├── conftest.py                 # Pytest fixtures and test utilities
├── test_emphy_size.py          # Tests for Gaussian LPF method
└── test_emphy_size_distance.py # Tests for distance transform method
```

**Fixtures (in `conftest.py`):**
- `synthetic_ct_volume` — Synthetic CT with lung and emphysema regions
- `synthetic_spacing` — Typical voxel spacing (2mm, 0.625mm, 0.625mm)
- `synthetic_lung_mask` — Lung segmentation mask
- `synthetic_emph_mask` — Emphysema (LAA) mask
- `temp_test_dir` — Temporary directory for test outputs

**Test Coverage:**
- Parameter validation and edge cases
- Core algorithm functions (sigma estimation, noise reduction, distance transform)
- Data container initialization and serialization
- Integration tests with synthetic data
- Output format and type correctness

## Usage

### Basic Pipeline (Gaussian LPF Method)

```python
from emphy_size import run_pipeline

# Load DICOM and classify
result = run_pipeline(
    dicom_dir="path/to/inspiratory_ct_dicom",
    dicom_dir_exp="path/to/expiratory_ct_dicom"  # optional
)

# Access results
print(result.summary())
print(f"  E1 volume: {result.e1_volume_ml:.1f} mL ({result.e1_fraction:.2f}%)")
print(f"  E2 volume: {result.e2_volume_ml:.1f} mL ({result.e2_fraction:.2f}%)")
print(f"  E3 volume: {result.e3_volume_ml:.1f} mL ({result.e3_fraction:.2f}%)")
print(f"  E4 volume: {result.e4_volume_ml:.1f} mL ({result.e4_fraction:.2f}%)")
```

### Distance Transform + Watershed Method

```python
from emphy_size_distance import run_pipeline

result = run_pipeline(
    dicom_dir="path/to/inspiratory_ct_dicom"
)

print(result.summary())
```

### Command Line

```bash
# Gaussian LPF method
python emphy_size.py

# Distance Transform + Watershed method
python emphy_size_distance.py
```

(Edit the `if __name__ == "__main__"` sections with your DICOM directory paths)

## Output

Both pipelines return an `EmphysemaResult` object containing:

```python
@dataclass
class EmphysemaResult:
    laa_percent: float           # Total %LAA
    
    e1_volume_ml: float          # E1 volume (mL)
    e2_volume_ml: float          # E2 volume (mL)
    e3_volume_ml: float          # E3 volume (mL)
    e4_volume_ml: float          # E4 volume (mL)
    
    e1_fraction: float           # E1 as % of lung volume
    e2_fraction: float           # E2 as % of lung volume
    e3_fraction: float           # E3 as % of lung volume
    e4_fraction: float           # E4 as % of lung volume
    
    lung_volume_ml: float        # Total lung volume (mL)
    voxel_size_ml: float         # Volume per voxel (mL)
    
    masks: dict                  # {'E1': mask, 'E2': mask, ...}
```

### Visualizations

Both pipelines include visualization helpers:

- **`visualize_orthogonal_views()`** — Show lung/emphysema masks on axial, coronal, sagittal slices
- **`visualize_distance_field()`** — Display Euclidean distance transform (distance-transform only)
- **`visualize_seeds_on_edt()`** — Overlay detected seeds on EDT (distance-transform only)
- **`visualize_subgroup_masks_on_image()`** — Color-coded emphysema subgroups on CT image
- **`visualize_subgroup_clusters_on_axial_slices()`** — Axial slice overlays at quartile heights
- **`visualize_lpf_iteration()`** — Gaussian filter output at each size iteration (Gaussian LPF only)

Enable via passing `volume` parameter and setting matplotlib to interactive mode:

```python
import matplotlib.pyplot as plt
plt.ion()  # interactive mode

result = run_pipeline("path/to/dicom")
# Visualizations display automatically during pipeline execution
```

## File Structure

```
emphysema_quantification/
├── emphy_size.py                    # Gaussian LPF-based classification (Oh et al. 2017)
├── emphy_size_distance.py           # Distance transform + watershed classification
├── emphy_size_notebook.ipynb        # Jupyter notebook: Gaussian LPF analysis
├── emphy_size_distance_notebook.ipynb  # Jupyter notebook: Distance transform analysis
├── README.md                        # This file
├── .gitignore                       # Python/Jupyter ignore rules
└── Figure_1.png                     # Reference diagram
```

## Comparison: Which Method to Use?

| Aspect | Gaussian LPF | Distance Transform |
|--------|------|-----------|
| **Published?** | Yes (Oh et al. 2017) | Novel alternative |
| **Parameters** | σ estimation required | Parameter-free |
| **Computational** | Iterative filtering | Single pass |
| **Non-spherical holes** | Assumes approximate spheres | Handles any shape |
| **Touching holes** | May underestimate | Naturally separated |
| **Direct interpretability** | Indirect (kernel size → size) | Direct (EDT value = radius) |

**Recommendation:**
- **Clinical validation needed:** Use Gaussian LPF (established method)
- **Research/algorithm comparison:** Use Distance Transform (novel, interpretable)
- **Best practice:** Run both and compare outputs

## Algorithm Details

### Gaussian LPF Method

The Gaussian LPF method applies iteratively scaled Gaussian filters to progressively segment emphysema by size:

1. **Gaussian convolution** with kernel sigma(gamma) = 0.147 + 2 * 0.1038 * gamma, where gamma is threshold diameter / 2
2. **Skeleton extraction** via local maxima (≥99.9% of peak)
3. **Anatomical dilation** to recover full hole boundary
4. **Subtraction** and repeat with smaller kernel

See `emphy_size.py` function `size_based_emphysema_clustering()` for details.

### Distance Transform + Watershed Method

EDT-based segmentation provides direct hole sizing:

1. **Distance transform**: Each voxel value = distance to nearest boundary (mm)
2. **Peak detection**: Local maxima with h-maxima suppression (1.0 mm)
3. **Watershed**: Region-growing from all peaks simultaneously
4. **Radius classification**: EDT value at seed = hole radius

See `emphy_size_distance.py` function `distance_transform_pipeline()` for details.

## References

1. **Oh et al. (2017)** — "Size variation and collapse of emphysema holes at inspiration and expiration CT scan"  
   *International Journal of COPD* 12:2043–2057  
   DOI: 10.2147/COPD.S130936

2. **Grady, L. (2006)** — "Random Walks for Image Segmentation"  
   *IEEE Transactions on Pattern Analysis and Machine Intelligence* 28(11):1768–1783  
   (Watershed algorithm foundation)

3. **Mehnert & Jackway (1997)** — "An improved seeded region growing algorithm"  
   *Pattern Recognition Letters* 18(10):1065–1071

## License

MIT License (adjust as needed)

## Author

Created for emphysema phenotyping research.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Submit a Pull Request

## Contact

For questions or bug reports, please open an issue on GitHub.
