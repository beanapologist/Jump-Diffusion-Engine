# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Root-level `pyproject.toml` for standard `pip install -e .` workflow.
- `tests/test_engine.py` — automated test suite covering `find_fixed_points`,
  `escape_probability`, `stationary_density`, `seat_and_release`, `simulate`,
  and `plot_trajectories`.
- `.github/workflows/ci.yml` — GitHub Actions CI running tests on Python 3.9–3.12.

### Fixed
- Implemented `plot_trajectories()`, which was previously a stub (`pass`).

### Changed
- README revised to use more precise, technically defensible language.
- `CITATION.cff` now includes the `version` field (`0.1.0`).

## [0.1.0] - TBD

### Added
- Initial public release of Jump-Diffusion-Engine.
- Core SDE simulator (`simulate`), fixed-point analysis (`find_fixed_points`,
  `basin_depth`, `potential`), escape probability estimation (`escape_probability`),
  stationary density (`stationary_density`), and basin-seating control
  (`seat_and_release`, `identify_boundary`).
- Packaging files under `packaging/` and root `pyproject.toml`.
