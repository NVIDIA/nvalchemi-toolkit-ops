# Changelog

## 0.2.0 - 2025-12-19

### Added

- Methods/kernels for computing electrostatic interactions
  - Includes direct Coulomb, Ewald, and particle mesh Ewald methods.
  - Some supporting math routines including spherical harmonics, spline
  evaluation, and Gaussian basis.
- New scripts in the `examples/electrostatics` folder that demonstrate
the new electrostatics interface.

### Changed

- Default behavior for `estimate_max_neighbors` is now more sensible
  - The default `atomic_density` value is changed from 0.5 to 0.35, which
  should provide better estimates of the maximum number of neighbors for
  most systems.
  - The rounding value has now been changed from the nearest power of 2
  to the nearest multiple of 16, which means the padding in neighbor
  matrices will be significantly lower and more realistic, as the prior
  behavior tended to significantly overpredict the maximum neighbor count.

### Fixed

- Issue #2 and #3 duplicate neighbors appearing in cell and batched cell lists.

## 0.1.0 - 2025-12-05

_First release of the package_

