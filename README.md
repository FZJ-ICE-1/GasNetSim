![](https://jugit.fz-juelich.de/iek-10/public/simulation/gasnetsim/-/raw/main/docs/GasNetSim_Logo.svg)

[![PyPI version](https://img.shields.io/pypi/v/GasNetSim.svg?color=orange)](https://pypi.org/project/GasNetSim/)
![Python Versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![DOI:10.1109/OSMSES54027.2022.9769148](https://zenodo.org/badge/DOI/10.1109/OSMSES54027.2022.9769148.svg)](https://doi.org/10.1109/OSMSES54027.2022.9769148)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fjugit.fz-juelich.de%2Fiek-10%2Fpublic%2Fsimulation%2Fgasnetsim/HEAD)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# **GasNetSim**

*GasNetSim* is a simulation package designed for gas network steady-state simulation.
It supports the steady-state natural gas network simulations with different gas mixture
compositions, thus enabling accurate analysis of the impacts of hydrogen injection on the gas network.  
Moreover, users have the flexibility to modify this tool and implement their own desired
gas mixture modeling approaches.

## Installation

Install from PyPI:

```bash
pip install GasNetSim
```

Or install from source. From the project root directory:

```bash
# Using uv (preferred, syncs all dependencies from the lockfile)
uv sync

# Using pip (editable install)
pip install -e .
```

For development (includes testing tools, Cantera, and Jupyter):

```bash
# Using uv
uv sync --group dev
```

## License

The project is released under the terms of the [MPL 2.0](https://mozilla.org/MPL/2.0/).

## Dependencies

### Runtime

<!-- Dependencies -->

- `numpy` >= 1.19.2, < 2.0.0
- `matplotlib` >= 3.3.2
- `scipy` >= 1.5.2
- `pandas` >= 1.1.3
- `requests` >= 2.25.1
- `pyparsing` ~= 3.0.7
- `tqdm` >= 4.64.1
- `seaborn` >= 0.12.2
- `numba` >= 0.58.1
- `plotly` >= 5.23.0
- `shapely` >= 2.0.6
- `geopandas` >= 1.0.1
- `networkx` == 3.2.1
- `pyarrow` >= 17.0.0
- `openpyxl` >= 3.1.5
- `cartopy` >= 0.24.1 (Python >= 3.10 only)
- `contextily` >= 1.6.2

<!-- End Dependencies -->

### Development / Testing

- `pytest` >= 8.0.0
- `black` >= 24.10.0
- `pre-commit` >= 4.0.1
- `parameterized` >= 0.9.0
- `cantera` ~= 3.0.0 — required for heating value comparison tests (`test_heating_value_comparison.py`)
- `jupyter` >= 1.1.1

## Reporting Issues

To report a problem, you can open an
[issue](https://jugit.fz-juelich.de/iek-10/public/simulation/gasnetsim/-/issues)
in the repository. If the issue is sensitive or security-related, please email
[Yifei Lu](yifei.lu@fz-juelich.de) directly.
