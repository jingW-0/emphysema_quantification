# Emphysema Quantification Pipeline

A Python framework for automated emphysema size classification from high-resolution CT scans. The repository includes three complementary algorithms for classifying emphysema low-attenuation regions into anatomical size groups.

## Size Classes

Emphysema is represented as air-filled low-attenuation regions within the lung. This project reports four size classes using diameter thresholds:

| Class | Diameter | Interpretation |
| --- | --- | --- |
| E1 | < 1.5 mm | Alveolar-level damage / smallest residual regions |
| E2 | 1.5-7 mm | Subacinar disease |
| E3 | 7-15 mm | Acinar or sublobular disease |
| E4 | >= 15 mm | Extra-lobular / largest holes |

The common preprocessing steps are DICOM loading, lung segmentation, emphysema mask extraction using HU < -950, and removal of connected components smaller than 2 voxels.

## Algorithms

### 1. Gaussian LPF Size Classification

File: `emphy_size.py`

This method follows the modified length scale method from Oh et al. (2017). It processes diameter thresholds from large to small, converts each diameter to a radius, estimates a Gaussian kernel, extracts high-response skeleton voxels, dilates by the anatomical radius, intersects with the original emphysema mask, and subtracts the classified region before the next iteration.

Key details:

- Diameter thresholds: 15.0, 7.0, 1.5 mm
- Radii used for filtering/dilation: 7.5, 3.5, 0.75 mm
- Sigma equation: `sigma = beta0 + 2 * beta1 * gamma`, where `gamma` is radius in mm
- Constants: `beta0 = 0.147`, `beta1 = 0.1038`

Best for: reproducing the published Oh et al. approach and comparing against an established method.

Tradeoffs: requires sigma estimation and Gaussian filtering can blur nearby holes or underestimate boundaries.

### 2. Distance Transform + Watershed

File: `emphy_size_distance.py`

This method computes the Euclidean distance transform (EDT) of the emphysema mask in physical millimeters. EDT peaks are treated as hole centers, h-maxima suppression selects seeds, and marker-controlled watershed separates regions. Each watershed region is classified by the EDT value at its seed.

Key details:

- EDT value at a peak approximates inscribed hole radius
- Default h-maxima threshold: 1.0 mm
- Radius thresholds: 0.75, 3.5, 7.5 mm
- Produces a per-hole catalogue with centroid, radius, diameter, subgroup, and volume

Best for: direct, interpretable per-hole segmentation when seed-based separation is useful.

Tradeoffs: watershed output depends on seed detection and can be sensitive in densely connected emphysema.

### 3. Iterative EDT Thresholding

File: `emphy_size_distance_iterative.py`

This deterministic method also uses the EDT, but avoids watershed seeds. It thresholds the EDT at anatomical radii from large to small, dilates each core by the same radius, intersects with the emphysema mask, and subtracts each classified region before processing smaller thresholds.

Key details:

- Radius thresholds: 7.5, 3.5, 0.75 mm
- Core voxels satisfy `EDT > radius`
- Output includes a `cluster_map` with labels 1-4 for E1-E4
- Main functions: `compute_edt()`, `edt_iterative_clustering()`, `compute_emphysema_indices()`

Best for: deterministic size classification without sigma estimation or seed tuning.

Tradeoffs: less instance-specific than watershed; classification follows threshold/dilation geometry.

## Method Comparison

| Aspect | Gaussian LPF | Distance + Watershed | Iterative EDT |
| --- | --- | --- | --- |
| Main file | `emphy_size.py` | `emphy_size_distance.py` | `emphy_size_distance_iterative.py` |
| Size signal | Gaussian kernel response | EDT peak radius | EDT threshold radius |
| Published basis | Oh et al. (2017) | Research alternative | Research alternative |
| Main parameter | Sigma equation | h-maxima seed threshold | Anatomical radius thresholds |
| Per-hole catalogue | No | Yes | Counts by cluster map |
| Touching holes | May merge/blur | Watershed separation | Iterative threshold/dilation |
| Interpretability | Indirect | Direct radius at seed | Direct radius threshold |

Recommendation:

- Use Gaussian LPF when reproducing the published method matters most.
- Use Distance + Watershed when per-hole instances and centers are important.
- Use Iterative EDT when you want a deterministic, geometry-based size map.
- For research comparisons, run all three and compare subgroup volumes/fractions.

## Installation

Requirements:

- Python 3.8+
- Anaconda/Miniconda recommended

Setup:

```bash
git clone https://github.com/jingW-0/emphysema_quantification.git
cd emphysema_quantification
conda create -n emphysema python=3.11
conda activate emphysema
pip install SimpleITK scipy numpy scikit-image matplotlib pydicom
```

For tests and development tools:

```bash
pip install -r requirements-dev.txt
```

## Usage

### Gaussian LPF

```python
from emphy_size import run_pipeline

result = run_pipeline(dicom_dir="path/to/inspiratory_ct_dicom")
print(result.summary())
```

### Distance Transform + Watershed

```python
from emphy_size_distance import run_pipeline

result = run_pipeline(dicom_dir="path/to/inspiratory_ct_dicom", h_mm=1.0)
print(result.summary())
print(result.hole_catalogue[:5])
```

### Iterative EDT Thresholding

```python
from emphy_size_distance_iterative import run_pipeline

result = run_pipeline(dicom_dir="path/to/inspiratory_ct_dicom")
print(result.summary())
cluster_map = result.cluster_map
```

### Command Line

Each script can be run directly after editing its `if __name__ == "__main__"` DICOM path:

```bash
python emphy_size.py
python emphy_size_distance.py
python emphy_size_distance_iterative.py
```

## Outputs

All three methods report:

- Total `%LAA`
- Lung volume in mL
- E1, E2, E3, and E4 volumes in mL
- E1, E2, E3, and E4 fractions as percent of lung volume

Method-specific outputs:

- `emphy_size.py`: subgroup masks in `result.masks`
- `emphy_size_distance.py`: subgroup masks plus `result.hole_catalogue`
- `emphy_size_distance_iterative.py`: labeled `result.cluster_map` and subgroup hole counts

## Visualization

Visualization helpers are available in the algorithm modules:

- `visualize_lpf_iteration()` in `emphy_size.py`
- `visualize_subgroup_clusters_on_axial_slices()` in `emphy_size.py`
- `visualize_orthogonal_views()` in `emphy_size_distance.py`
- `visualize_distance_field()` in `emphy_size_distance.py`
- `visualize_seeds_on_edt()` in `emphy_size_distance.py`
- `visualize_subgroup_masks_on_image()` in `emphy_size_distance.py`

The pipeline scripts currently display several matplotlib figures during execution.

## Examples

The iterative EDT example notebook builds a synthetic CT example, runs the iterative EDT thresholding method, and visualizes the resulting cluster map.

- Notebook: `emphy_size_distance_iterative_example.ipynb`

Run it with:

```bash
jupyter notebook emphy_size_distance_iterative_example.ipynb
```

## Testing

Run all tests:

```bash
pytest
```

Run individual test files:

```bash
pytest tests/test_emphy_size.py -v
pytest tests/test_emphy_size_distance.py -v
pytest tests/test_emphy_size_distance_iterative.py -v
```

Run coverage:

```bash
pytest --cov=. --cov-report=html
```

Run formatting and checks:

```bash
black emphy_size.py emphy_size_distance.py emphy_size_distance_iterative.py tests/
isort emphy_size.py emphy_size_distance.py emphy_size_distance_iterative.py tests/
flake8 emphy_size.py emphy_size_distance.py emphy_size_distance_iterative.py tests/
mypy emphy_size.py emphy_size_distance.py emphy_size_distance_iterative.py --ignore-missing-imports
```

## Repository Structure

```text
emphysema_quantification/
|-- emphy_size.py                         # Gaussian LPF method
|-- emphy_size_distance.py                # Distance transform + watershed method
|-- emphy_size_distance_iterative.py      # Iterative EDT thresholding method
|-- emphy_size_distance_iterative_example.ipynb
|-- tests/
|   |-- test_emphy_size.py
|   |-- test_emphy_size_distance.py
|   |-- test_emphy_size_distance_iterative.py
|   `-- conftest.py
|-- requirements-dev.txt
|-- pytest.ini
|-- tox.ini
|-- Figure_1.png
`-- README.md
```

## References

1. Oh et al. (2017), "Size variation and collapse of emphysema holes at inspiration and expiration CT scan", International Journal of COPD 12:2043-2057. DOI: 10.2147/COPD.S130936
2. Grady, L. (2006), "Random Walks for Image Segmentation", IEEE Transactions on Pattern Analysis and Machine Intelligence 28(11):1768-1783.
3. Mehnert & Jackway (1997), "An improved seeded region growing algorithm", Pattern Recognition Letters 18(10):1065-1071.

## License

MIT License.

## Contact

For questions or bug reports, please open an issue on GitHub.
