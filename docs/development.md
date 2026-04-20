# Development Guide

This guide provides development information for contributors working on the GasNetSim project.

## Development Commands

### Installation & Setup
```bash
# Install in editable mode
pip install -e .
pip install -r requirements.txt

# Or using Poetry (preferred)
poetry install
poetry shell
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_network.py
pytest tests/test_gerg2008_eos.py
pytest tests/test_timeseries.py
```

### Code Quality
```bash
# Format code (automatically enforced)
black .

# Setup/run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Build & Dependencies
```bash
# Export requirements.txt from pyproject.toml (done automatically by pre-commit)
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Architecture Overview

**GasNetSim** is a gas network steady-state simulation package with hydrogen injection analysis capabilities.

### Core Components (`GasNetSim/components/`)

- **Network** (`network.py`) - Main simulation engine using sparse matrix operations, optional CUDA support
- **Node** (`node.py`) - Network nodes (demand/supply/junction) with gas composition and geographic data
- **Pipeline** (`pipeline.py`) - Gas transmission pipelines with flow calculations and batch tracking
- **Gas Mixture** (`gas_mixture/`) - Advanced property calculations via GERG-2008 equation of state
  - GERG-2008 implementation with Numba optimization for performance
  - Viscosity methods (Herning-Zipperer, Lucas)
  - Heating value calculations
- **Utils** (`utils/`) - Network creation from CSV, CUDA support, plotting functions

### Simulation Engine (`simulation/`)
- **Time Series** (`timeseries.py`) - Dynamic simulations with profile-based demand modeling

### Key Dependencies
- **Scientific**: NumPy, SciPy, Pandas, Matplotlib
- **Specialized**: Cantera (chemical properties), NetworkX (graph operations)
- **Performance**: Numba (JIT compilation), optional CuPy (GPU acceleration)
- **Geospatial**: GeoPandas, Shapely (network visualization)

## Network Definition Pattern

Networks are typically defined using CSV files:
- `*_nodes.csv` - Node definitions with coordinates, types, and properties
- `*_pipelines.csv` - Pipeline connections with physical properties
- `*_resistance.csv` - Network resistance data (optional)

Example usage:
```python
import GasNetSim as gns
from pathlib import Path

network = gns.create_network_from_csv(Path("./"))
network.simulation(use_cuda=False, tol=0.0001)
```

## Testing Approach

- **Validation tests** comparing against reference implementations
- **Performance benchmarks** (Numba vs standard implementations) 
- **Property calculation accuracy** tests for GERG-2008 and heating values
- **Network simulation** tests using example networks (Irish13, German_resistance)
- **Jupyter notebooks** for visual validation in tests/ directory

## Recent Improvements

**Sparse Index Mapping Support (2025):**
- Fixed IndexError crashes when networks have non-consecutive node indices (e.g., [1, 5, 10, 15])
- Added index mapping system in Network class to handle sparse → dense matrix index conversion
- Updated all matrix operations to use proper indexing
- Maintained backward compatibility with existing networks
- Comprehensive test coverage for sparse indexing scenarios

## Current Development Areas

Based on recent commits and git status:
- GERG-2008 equation of state improvements
- Time series simulation capabilities  
- Pipeline modeling enhancements
- Tutorial development and documentation