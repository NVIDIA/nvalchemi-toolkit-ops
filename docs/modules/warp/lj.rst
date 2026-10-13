:mod:`nvalchemiops.interactions`: Lennard-Jones
===============================================

.. currentmodule:: nvalchemiops.interactions

Warp launchers for the Lennard-Jones pair potential. Each launcher accepts a
neighbor matrix or a CSR neighbor list. A nonzero ``switch_width`` applies the
C2 switching function below over the last ``switch_width`` of the cutoff.

.. autofunction:: lj_energy
.. autofunction:: lj_forces
.. autofunction:: lj_energy_forces
.. autofunction:: lj_energy_forces_virial

Switching Function
------------------

.. autofunction:: switch_c2
