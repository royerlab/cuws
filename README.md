# CuWS

GPU-accelerated watershed segmentation from minima using CuPy.

This implements only the watershed from minima algorithm, not the seeded watershed.

## Installation

```bash
pip install cuws
```

## Usage

```python
from cuws import watershed_from_minima

...

labels = watershed_from_minima(image, mask)
```

## Requirements

- Python >= 3.11
- CUDA-compatible GPU
- CuPy
