:mod:`nvalchemiops.dynamics`: Molecular Dynamics
===================================================

.. automodule:: nvalchemiops.dynamics
    :no-members:
    :no-inherited-members:

Warp-Level Interface
--------------------

.. tip::
   This is the low-level Warp interface that operates on ``warp.array`` objects.
   For PyTorch tensor support with the FIRE2 optimizer, see :doc:`../torch/dynamics`.

High-Level Interface
~~~~~~~~~~~~~~~~~~~~

This module provides GPU-accelerated integrators and thermostats for molecular dynamics
simulations. All functions support automatic differentiation through PyTorch's autograd
system. Most functions offer three execution modes: single system, batched with
``batch_idx``, and batched with ``atom_ptr``. NPT/NPH integrators support
``batch_idx`` only -- see individual function docs for details.

.. tip::
    Check out the :ref:`dynamics_userguide` page for usage examples and
    a conceptual overview of the available integrators and thermostats.

Integrators
-----------

Velocity Verlet
~~~~~~~~~~~~~~~

Time-reversible, symplectic integrator for NVE (microcanonical) ensemble.

.. autofunction:: nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_position_update
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_velocity_finalize
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_position_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_velocity_finalize_out

Langevin Dynamics
~~~~~~~~~~~~~~~~~

BAOAB splitting scheme for NVT (canonical) ensemble with stochastic thermostat.

.. autofunction:: nvalchemiops.dynamics.integrators.langevin.langevin_baoab_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.langevin.langevin_baoab_finalize
.. autofunction:: nvalchemiops.dynamics.integrators.langevin.langevin_baoab_half_step_out
.. autofunction:: nvalchemiops.dynamics.integrators.langevin.langevin_baoab_finalize_out

Nosé-Hoover Chain
~~~~~~~~~~~~~~~~~~

Deterministic thermostat for NVT ensemble using Nosé-Hoover chains with Yoshida-Suzuki integration.

.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_velocity_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_position_update
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_thermostat_chain_update
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_velocity_half_step_out
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_position_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_thermostat_chain_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_compute_masses
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_compute_chain_energy

Velocity Rescaling
~~~~~~~~~~~~~~~~~~

Simple velocity rescaling thermostat for quick equilibration (non-canonical).

.. autofunction:: nvalchemiops.dynamics.integrators.velocity_rescaling.velocity_rescale
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_rescaling.velocity_rescale_out
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_rescaling.compute_rescale_factor

NPT/NPH (Isothermal-Isobaric / Isenthalpic-Isobaric)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extended ensemble integrators for constant pressure simulations using Nosé-Hoover chains
and Martyna-Tobias-Klein barostat. Supports both isotropic and anisotropic pressure control.

**Pressure Calculations**

.. autofunction:: nvalchemiops.dynamics.integrators.npt.compute_pressure_tensor
.. autofunction:: nvalchemiops.dynamics.integrators.npt.compute_scalar_pressure

**NPT Integration (Isothermal-Isobaric)**

.. autofunction:: nvalchemiops.dynamics.integrators.npt.npt_thermostat_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.npt.npt_barostat_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.npt.npt_velocity_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.npt.npt_position_update
.. autofunction:: nvalchemiops.dynamics.integrators.npt.npt_cell_update
.. autofunction:: nvalchemiops.dynamics.integrators.npt.npt_velocity_half_step_out
.. autofunction:: nvalchemiops.dynamics.integrators.npt.npt_position_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.npt.npt_cell_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.npt.run_npt_step

**NPH Integration (Isenthalpic-Isobaric)**

.. autofunction:: nvalchemiops.dynamics.integrators.npt.nph_barostat_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.npt.nph_velocity_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.npt.nph_position_update
.. autofunction:: nvalchemiops.dynamics.integrators.npt.nph_cell_update
.. autofunction:: nvalchemiops.dynamics.integrators.npt.nph_velocity_half_step_out
.. autofunction:: nvalchemiops.dynamics.integrators.npt.nph_position_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.npt.run_nph_step

**Barostat Utilities**

.. autofunction:: nvalchemiops.dynamics.integrators.npt.compute_barostat_mass
.. autofunction:: nvalchemiops.dynamics.integrators.npt.compute_cell_kinetic_energy
.. autofunction:: nvalchemiops.dynamics.integrators.npt.compute_barostat_potential_energy

Optimizers
----------

FIRE
~~~~

Fast Inertial Relaxation Engine for geometry optimization.

.. autofunction:: nvalchemiops.dynamics.optimizers.fire.fire_step
.. autofunction:: nvalchemiops.dynamics.optimizers.fire.fire_update

FIRE2
~~~~~

Improved FIRE optimizer with adaptive damping and velocity mixing.

.. autofunction:: nvalchemiops.dynamics.optimizers.fire2.fire2_step

Thermostat Utilities
--------------------

Functions for temperature control and velocity initialization.

.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.compute_kinetic_energy
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.compute_temperature
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.initialize_velocities
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.initialize_velocities_out
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.remove_com_motion
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.remove_com_motion_out

Cell Utilities
--------------

Functions for periodic cell operations and coordinate transformations.

.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.compute_cell_volume
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.compute_cell_inverse
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.compute_strain_tensor
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.apply_strain_to_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.scale_positions_with_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.scale_positions_with_cell_out
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.wrap_positions_to_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.wrap_positions_to_cell_out
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.cartesian_to_fractional
.. autofunction:: nvalchemiops.dynamics.utils.cell_utils.fractional_to_cartesian

Cell Filter Utilities (Variable-Cell Optimization)
--------------------------------------------------

Utilities for packing/unpacking extended arrays for variable-cell optimization.

.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.align_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.extend_batch_idx
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.extend_atom_ptr
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.pack_positions_with_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.pack_velocities_with_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.pack_forces_with_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.pack_masses_with_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.unpack_positions_with_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.unpack_velocities_with_cell
.. autofunction:: nvalchemiops.dynamics.utils.cell_filter.stress_to_cell_force

Constraint Utilities (SHAKE/RATTLE)
-----------------------------------

Holonomic constraint algorithms for bond length constraints.

.. autofunction:: nvalchemiops.dynamics.utils.constraints.shake_constraints
.. autofunction:: nvalchemiops.dynamics.utils.constraints.shake_iteration
.. autofunction:: nvalchemiops.dynamics.utils.constraints.shake_constraints_out
.. autofunction:: nvalchemiops.dynamics.utils.constraints.shake_iteration_out
.. autofunction:: nvalchemiops.dynamics.utils.constraints.rattle_constraints
.. autofunction:: nvalchemiops.dynamics.utils.constraints.rattle_iteration
.. autofunction:: nvalchemiops.dynamics.utils.constraints.rattle_constraints_out
.. autofunction:: nvalchemiops.dynamics.utils.constraints.rattle_iteration_out
