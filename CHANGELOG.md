# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.1

### Changed
- Use caches for gas mixture properties calculation to accelerate imports
- Update tutorials/02_gas_mixture_properties.ipynb to show the import speed improvement

## v0.2.0 - 2025-12-27

### Added

- Tutorials implemented using Binder
- Function to create network using files
- Use `networkx` for topological sort of pipelines and nodes
- Different reference temperature values for gas mixture heating value calculation (15 or 25 degrees Celsius)
- Reference pressure, temperature and composition for gas mixture initialization
- pre-commit hooks for automatically updating requirements.txt
- Compressor model and corresponding examples and tests

### Changed

- Changed the packaging tool from `setuptools` / `poetry` to `uv`
- Changed reference temperature for GasMixture class (wrt. density and combustion properties)
- Changed run_time_series() to use profiles instead of profile file path
- Improved results saving of time series simulations (support different output file formats)
- Simulation solver uses individual indexing system for nodes and pipelines

### Deprecated
- Renamed create_network_from_csv() to create_network_from_folder() for clarity and add FutureWarning

## v0.1.0 - 2024-09-10

### Added

- Added support for parallel pipelines
- Test for pipeline outlet temperature calculation function
- Test and validation with original [C++ implementation](https://github.com/usnistgov/AGA8/tree/master/AGA8CODE/C) of
  GERG-2008 EOS
- Added support of heating value calculation for GERG2008 gas mixture class
- Tests and validations for the implementation of the heating values calculation
- Tests and validations of gas mixture properties calculation using the _thermo_ package and the GERG-2008 EOS

### Changed

- Change `Cantera` dependency to `v3.0.0`
- Fixed pipeline outlet temperature calculation
- Fixed GERG-2008 implementation
- Set default gas mixture EOS for simulation to GERG-2008

### Removed

- Local adapted implementation of `thermo` package