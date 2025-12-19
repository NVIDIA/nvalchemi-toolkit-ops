:mod:`nvalchemiops.interactions.electrostatics`: Electrostatic Interactions
===========================================================================

.. automodule:: nvalchemiops.interactions.electrostatics
    :no-members:
    :no-inherited-members:

High-Level Interface
--------------------

These functions provide the end-user facing functions for computing electrostatic
interactions using Ewald summation, Particle Mesh Ewald (PME), and direct Coulomb
methods. All functions support automatic differentiation through PyTorch's
autograd system.

.. tip::
    Check out the :ref:`electrostatics_userguide` page for usage examples and
    a conceptual overview of the available electrostatic methods.

Ewald Summation
^^^^^^^^^^^^^^^

Complete Ewald summation combining real-space and reciprocal-space contributions.
Supports both single-system and batched calculations via the ``batch_idx`` parameter.

.. autofunction:: nvalchemiops.interactions.electrostatics.ewald.ewald_summation
.. autofunction:: nvalchemiops.interactions.electrostatics.ewald.ewald_real_space
.. autofunction:: nvalchemiops.interactions.electrostatics.ewald.ewald_reciprocal_space

Particle Mesh Ewald (PME)
^^^^^^^^^^^^^^^^^^^^^^^^^

FFT-based Ewald method achieving :math:`O(N \log N)` scaling. Uses B-spline interpolation
for efficient charge assignment and force interpolation.

.. autofunction:: nvalchemiops.interactions.electrostatics.pme.particle_mesh_ewald
.. autofunction:: nvalchemiops.interactions.electrostatics.pme.pme_reciprocal_space

Direct Coulomb
^^^^^^^^^^^^^^

Direct pairwise Coulomb interactions, supporting both undamped (1/r) and damped
:math:`(\text{erfc}(\alpha r)/r)` variants. Useful for isolated systems or as
the real-space component of Ewald/PME.

.. autofunction:: nvalchemiops.interactions.electrostatics.coulomb.coulomb_energy
.. autofunction:: nvalchemiops.interactions.electrostatics.coulomb.coulomb_forces
.. autofunction:: nvalchemiops.interactions.electrostatics.coulomb.coulomb_energy_forces

Parameter Estimation
--------------------

Functions for automatic parameter estimation based on desired accuracy tolerance.

Ewald Parameters
^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.interactions.electrostatics.parameters.estimate_ewald_parameters
.. autoclass:: nvalchemiops.interactions.electrostatics.parameters.EwaldParameters
    :members:
    :undoc-members:

PME Parameters
^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.interactions.electrostatics.parameters.estimate_pme_parameters
.. autofunction:: nvalchemiops.interactions.electrostatics.parameters.estimate_pme_mesh_dimensions
.. autofunction:: nvalchemiops.interactions.electrostatics.parameters.mesh_spacing_to_dimensions
.. autoclass:: nvalchemiops.interactions.electrostatics.parameters.PMEParameters
    :members:
    :undoc-members:

Utility Functions
-----------------

K-Vector Generation
^^^^^^^^^^^^^^^^^^^

Functions for generating reciprocal-space vectors for Ewald summation and PME.

.. autofunction:: nvalchemiops.interactions.electrostatics.k_vectors.generate_k_vectors_ewald_summation
.. autofunction:: nvalchemiops.interactions.electrostatics.k_vectors.generate_k_vectors_pme

B-Spline Functions
^^^^^^^^^^^^^^^^^^

Functions for B-spline charge spreading and gathering, used by PME.

.. autofunction:: nvalchemiops.spline.spline_spread
.. autofunction:: nvalchemiops.spline.spline_gather
.. autofunction:: nvalchemiops.spline.spline_gather_vec3
.. autofunction:: nvalchemiops.spline.spline_gather_gradient
