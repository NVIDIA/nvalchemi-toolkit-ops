:mod:`nvalchemiops.torch.autograd`: Autograd Utilities
======================================================

.. currentmodule:: nvalchemiops.torch.autograd

.. automodule:: nvalchemiops.torch.autograd
    :no-members:
    :no-inherited-members:

Custom Op Registration
----------------------

.. autofunction:: warp_custom_op
.. autoclass:: OutputSpec
   :members:

Warp-PyTorch Interop
---------------------

.. autofunction:: warp_stream_from_torch
.. autofunction:: warp_from_torch
.. autofunction:: needs_grad

Autograd Context Manager
-------------------------

.. autoclass:: WarpAutogradContextManager
   :members:

.. autofunction:: attach_for_backward
.. autofunction:: retrieve_for_backward
.. autofunction:: extract_gradients
.. autofunction:: standard_backward
