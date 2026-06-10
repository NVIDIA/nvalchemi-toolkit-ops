# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Packed ``multipole_moments`` <-> per-:math:`l` Cartesian channels.

The public multipole-electrostatics bindings accept a single packed
``multipole_moments`` tensor of shape ``(N, (l_max + 1)**2)`` in e3nn
real-spherical-harmonic order:

==============  =====  ==============================================
slice           ``l``  contents
==============  =====  ==============================================
``[:, 0]``      0      charge
``[:, 1:4]``    1      dipole, e3nn ``(y, z, x)`` order (= raw Cartesian
                       dipole, permuted) — matches the customer reference.
``[:, 4:9]``    2      quadrupole, e3nn ``component``-normalized real
                       spherical-harmonic coefficients (**traceless**, 5).
==============  =====  ==============================================

The internal Warp kernels are Cartesian: scalar ``charges``, ``(x, y, z)``
``dipoles``, and a symmetric **traceless** ``(N, 3, 3)`` ``quadrupoles``.
This module converts between the two layouts at the wrapper boundary with
pure ``torch`` ops (static-shape gather / matmul / stack), so the whole
conversion stays on the autograd tape and is ``torch.compile`` /
CUDA-graph friendly.

l=2 convention
--------------
The l=2 channel is the e3nn ``component``-normalized real-SH basis (what an
e3nn irrep head emits). The conversion is fixed by requiring

.. math::

    \hat k \cdot Q \cdot \hat k = \sum_m c_m\, Y_{2m}(\hat k);

the constant matrices below are the exact closed form. The l=2 channel is
traceless: the isotropic trace of a Cartesian quadrupole is dropped on the
way in (it is unrepresentable in 5 spherical components).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "infer_l_max",
    "split_multipole_moments",
    "split_packed_for_kernels",
    "split_source_feats",
    "pack_multipole_moments",
    "pack_charges_dipoles",
    "e3nn_to_cartesian_quadrupole",
    "cartesian_quadrupole_to_e3nn",
    "dipole_spherical_to_cartesian",
    "dipole_cartesian_to_spherical",
]

# Dipole bare (N, 3): e3nn (y, z, x) <-> Cartesian (x, y, z); inverse perms.
_DIP_CART_TO_SPH = (1, 2, 0)
_DIP_SPH_TO_CART = (2, 0, 1)
# Packed (N, 4) [charge, e3nn dipole]: gather Cartesian (x, y, z) from slots 3,1,2.
_DIP_PACKED_SPH_TO_CART = (3, 1, 2)

_S5 = math.sqrt(5.0)
_S15 = math.sqrt(15.0)
_HALF_S5 = _S5 / 2.0
_HALF_S15 = _S15 / 2.0

# q6 = B @ feats5.
_B_E3NN_TO_CART = (
    (0.0, 0.0, -_HALF_S5, 0.0, -_HALF_S15),  # Qxx
    (0.0, 0.0, _S5, 0.0, 0.0),  # Qyy
    (0.0, 0.0, -_HALF_S5, 0.0, _HALF_S15),  # Qzz
    (0.0, _HALF_S15, 0.0, 0.0, 0.0),  # Qxy
    (_HALF_S15, 0.0, 0.0, 0.0, 0.0),  # Qxz
    (0.0, 0.0, 0.0, _HALF_S15, 0.0),  # Qyz
)

# Symmetric (3, 3) <-> q6 gather index: mat[i, j] = q6[_Q6_OF_MAT[i, j]].
_Q6_OF_MAT = ((0, 3, 4), (3, 1, 5), (4, 5, 2))

# Per-(dtype, device-string) cached constant tensors (built once).
_CACHE: dict = {}


def _consts(dtype: torch.dtype, device: torch.device):
    """Return ``(B, T, q6_idx)`` constant tensors for ``(dtype, device)``."""
    key = (dtype, str(device))
    cached = _CACHE.get(key)
    if cached is None:
        B = torch.tensor(_B_E3NN_TO_CART, dtype=dtype, device=device)  # (6, 5)
        T = torch.linalg.pinv(B.to(torch.float64)).to(dtype)  # (5, 6)
        idx = torch.tensor(_Q6_OF_MAT, dtype=torch.long, device=device)  # (3, 3)
        cached = (B, T, idx)
        _CACHE[key] = cached
    return cached


def infer_l_max(multipole_moments: torch.Tensor) -> int:
    """Return ``l_max`` implied by ``multipole_moments.shape[-1]``.

    =============  =======
    last-dim size  l_max
    =============  =======
    1              0   (charges)
    4              1   (charges + dipoles)
    9              2   (charges + dipoles + quadrupoles)
    =============  =======
    """
    if multipole_moments.ndim != 2:
        raise ValueError(
            f"multipole_moments must be rank-2 (N, (l_max+1)^2), got shape "
            f"{tuple(multipole_moments.shape)}."
        )
    last = multipole_moments.shape[-1]
    sizes = {1: 0, 4: 1, 9: 2}
    if last not in sizes:
        raise ValueError(
            "multipole_moments last-dim must be 1 (l_max=0), 4 (l_max=1), or "
            f"9 (l_max=2); got {last}."
        )
    return sizes[last]


def dipole_spherical_to_cartesian(dipole_sph: torch.Tensor) -> torch.Tensor:
    """Permute an e3nn ``(y, z, x)`` dipole to Cartesian ``(x, y, z)``.

    Parameters
    ----------
    dipole_sph : torch.Tensor, shape ``(..., 3)``
        Dipole in e3nn spherical ``(y, z, x)`` order.

    Returns
    -------
    torch.Tensor, shape ``(..., 3)``
        Same data permuted to Cartesian ``(x, y, z)`` (contiguous).
    """
    return dipole_sph[..., _DIP_SPH_TO_CART].contiguous()


def dipole_cartesian_to_spherical(dipole_cart: torch.Tensor) -> torch.Tensor:
    """Permute a Cartesian ``(x, y, z)`` dipole to e3nn ``(y, z, x)`` order.

    Parameters
    ----------
    dipole_cart : torch.Tensor, shape ``(..., 3)``
        Dipole in Cartesian ``(x, y, z)`` order.

    Returns
    -------
    torch.Tensor, shape ``(..., 3)``
        Same data permuted to e3nn ``(y, z, x)`` (contiguous).
    """
    return dipole_cart[..., _DIP_CART_TO_SPH].contiguous()


def e3nn_to_cartesian_quadrupole(feats5: torch.Tensor) -> torch.Tensor:
    """Convert e3nn l=2 coefficients to a symmetric traceless Cartesian tensor.

    Pure torch (matmul + gather); autograd- and graph-friendly. The output is
    traceless by construction.

    Parameters
    ----------
    feats5 : torch.Tensor, shape ``(N, 5)``
        e3nn ``component``-normalized real-SH l=2 coefficients
        (``m = -2, -1, 0, +1, +2`` order).

    Returns
    -------
    torch.Tensor, shape ``(N, 3, 3)``
        Symmetric traceless Cartesian quadrupole tensor.
    """
    B, _, idx = _consts(feats5.dtype, feats5.device)
    q6 = feats5 @ B.t()  # (N, 6) = [xx, yy, zz, xy, xz, yz]
    return q6[:, idx]  # (N, 3, 3) symmetric


def cartesian_quadrupole_to_e3nn(quadrupoles: torch.Tensor) -> torch.Tensor:
    """Convert a symmetric Cartesian quadrupole to e3nn l=2 coefficients.

    The map annihilates the isotropic trace (l=2 is traceless), so any trace
    in the input is silently dropped — callers that care should detrace first
    (``pack_multipole_moments`` warns).

    Parameters
    ----------
    quadrupoles : torch.Tensor, shape ``(N, 3, 3)``
        Symmetric Cartesian quadrupole tensor (defensively symmetrized).

    Returns
    -------
    torch.Tensor, shape ``(N, 5)``
        e3nn ``component``-normalized real-SH l=2 coefficients.
    """
    _, T, _ = _consts(quadrupoles.dtype, quadrupoles.device)
    # Symmetrize defensively, then pull the 6 unique entries.
    q = 0.5 * (quadrupoles + quadrupoles.transpose(-1, -2))
    q6 = torch.stack(
        [q[:, 0, 0], q[:, 1, 1], q[:, 2, 2], q[:, 0, 1], q[:, 0, 2], q[:, 1, 2]],
        dim=-1,
    )
    return q6 @ T.t()  # (N, 5)


def split_multipole_moments(
    multipole_moments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int]:
    """Slice packed moments into ``(charges, dipoles_cart, quadrupoles_cart, l_max)``.

    ``dipoles_cart`` is ``None`` for l_max=0; ``quadrupoles_cart`` is ``None``
    for l_max<2. All outputs stay on the autograd graph (gradients route back
    through the converter to the packed input).

    Parameters
    ----------
    multipole_moments : torch.Tensor, shape ``(N, (l_max+1)**2)``
        Packed e3nn moments (last-dim 1, 4, or 9).

    Returns
    -------
    charges : torch.Tensor, shape ``(N,)``
    dipoles_cart : torch.Tensor or None, shape ``(N, 3)``
        Cartesian ``(x, y, z)`` dipole; ``None`` for ``l_max == 0``.
    quadrupoles_cart : torch.Tensor or None, shape ``(N, 3, 3)``
        Symmetric traceless Cartesian quadrupole; ``None`` for ``l_max < 2``.
    l_max : int
        Inferred angular order (0, 1, or 2).
    """
    l_max = infer_l_max(multipole_moments)
    charges = multipole_moments[:, 0]
    dipoles_cart = None
    quadrupoles_cart = None
    if l_max >= 1:
        dipoles_cart = multipole_moments[:, _DIP_PACKED_SPH_TO_CART].contiguous()
    if l_max >= 2:
        quadrupoles_cart = e3nn_to_cartesian_quadrupole(multipole_moments[:, 4:9])
    return charges, dipoles_cart, quadrupoles_cart, l_max


def split_packed_for_kernels(
    multipole_moments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Return ``(source_feats_l1, quadrupoles_cart, l_max)`` for the Ewald/PME
    internal contract.

    ``source_feats_l1`` is the e3nn l<=1 block ``(N, 1)`` or ``(N, 4)`` — the
    legacy packed layout the internal kernels / self-energy / SCF-step consume
    unchanged. The l=2 block (when present) is converted to a Cartesian
    symmetric **traceless** ``(N, 3, 3)``. Both stay on the autograd graph, so
    gradients route back through the converter to ``multipole_moments``.
    """
    l_max = infer_l_max(multipole_moments)
    n_l1 = 4 if l_max >= 1 else 1
    source_feats_l1 = multipole_moments[:, :n_l1].contiguous()
    quadrupoles_cart = (
        e3nn_to_cartesian_quadrupole(multipole_moments[:, 4:9]) if l_max == 2 else None
    )
    return source_feats_l1, quadrupoles_cart, l_max


def split_source_feats(
    source_feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Slice the l<=1 packed block into ``(charges, dipoles_cart, l_max)``.

    Kernel-boundary helper for the l<=1 charge+dipole packed block (``(N, 1)``
    or ``(N, 4)`` in e3nn ``[q, μ_y, μ_z, μ_x]`` order) that the Warp
    kernels / SCF-step / self-energy consume. Equivalent to the l<=1 prefix
    of :func:`split_multipole_moments` (no quadrupole channel). All outputs
    stay on the autograd graph.

    Parameters
    ----------
    source_feats : torch.Tensor, shape ``(N, 1)`` or ``(N, 4)``
        Packed l<=1 e3nn block (``[q]`` or ``[q, μ_y, μ_z, μ_x]``).

    Returns
    -------
    charges : torch.Tensor, shape ``(N,)``
    dipoles_cart : torch.Tensor or None, shape ``(N, 3)``
        Cartesian ``(x, y, z)`` dipole; ``None`` for ``l_max == 0``.
    l_max : int
        Inferred angular order (0 or 1).
    """
    l_max = infer_l_max(source_feats)
    charges = source_feats[..., 0]
    if l_max == 0:
        return charges, None, 0
    dipoles_cart = source_feats[..., _DIP_PACKED_SPH_TO_CART].contiguous()
    return charges, dipoles_cart, l_max


def pack_charges_dipoles(
    charges: torch.Tensor, dipoles_cart: torch.Tensor | None
) -> torch.Tensor:
    """Inverse of :func:`split_source_feats` — build the l<=1 packed block.

    Concatenates ``charges[..., None]`` with the e3nn-permuted
    ``dipoles_cart``; ``dipoles_cart=None`` yields shape ``(N, 1)``. Round-trip
    helper for tests and the l<=1 internal contract.
    """
    if dipoles_cart is None:
        return charges.unsqueeze(-1).contiguous()
    return torch.cat(
        [charges.unsqueeze(-1), dipole_cartesian_to_spherical(dipoles_cart)], dim=-1
    ).contiguous()


def pack_multipole_moments(
    charges: torch.Tensor,
    dipoles: torch.Tensor | None = None,
    quadrupoles: torch.Tensor | None = None,
    *,
    trace_atol: float = 1e-8,
) -> torch.Tensor:
    """Build packed e3nn ``multipole_moments`` from Cartesian channels.

    Parameters
    ----------
    charges : ``(N,)``
    dipoles : ``(N, 3)`` Cartesian ``(x, y, z)`` or ``None``.
    quadrupoles : ``(N, 3, 3)`` symmetric Cartesian or ``None``. The l=2
        channel is traceless; a non-negligible input trace (> ``trace_atol``)
        is dropped and raises a warning.
    trace_atol : float, default 1e-8
        Absolute tolerance on ``max |Tr Q|`` above which the dropped
        quadrupole trace triggers a warning.

    Returns
    -------
    torch.Tensor, shape ``(N, (l_max+1)**2)``
        Packed e3nn ``multipole_moments`` (last-dim 1, 4, or 9 depending on
        which channels were supplied), contiguous, on ``charges.device``.

    Raises
    ------
    ValueError
        If ``quadrupoles`` is given without ``dipoles`` (the packed l_max=2
        layout requires the ``(N, 4)`` charge+dipole block).
    """
    # Input shape validation. This is a host-side constructor called once to
    # build inputs (not on the torch.compile hot path), so cheap Python checks
    # are safe and catch mis-shaped moments before they reach the kernels.
    if charges.ndim != 1:
        raise ValueError(f"charges must be 1-D (N,); got {tuple(charges.shape)}")
    n = charges.shape[0]
    if dipoles is not None and dipoles.shape != (n, 3):
        raise ValueError(
            f"dipoles must be (N, 3) = ({n}, 3); got {tuple(dipoles.shape)}"
        )
    if quadrupoles is not None and quadrupoles.shape != (n, 3, 3):
        raise ValueError(
            f"quadrupoles must be (N, 3, 3) = ({n}, 3, 3); "
            f"got {tuple(quadrupoles.shape)}"
        )
    cols = [charges.reshape(n, 1)]
    if dipoles is not None:
        cols.append(dipole_cartesian_to_spherical(dipoles))  # (N, 3) e3nn order
    if quadrupoles is not None:
        if dipoles is None:
            raise ValueError(
                "quadrupoles given without dipoles: packed l_max=2 moments "
                "require the (N, 4) charge+dipole block. Pass dipoles "
                "(use zeros for a pure quadrupole)."
            )
        # Diagnostic only: warn if Q is not traceless. Skipped under
        # ``torch.compile`` because ``float(...)`` is a device sync (a graph
        # break on the hot path); ``cartesian_quadrupole_to_e3nn`` below drops
        # the trace regardless of this check. Detach so a grad-requiring Q does
        # not emit a spurious "requires_grad to a scalar" warning.
        if not torch.compiler.is_compiling():
            tr_max = float(
                quadrupoles.detach().diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()
            )
            if tr_max > trace_atol:
                import warnings

                warnings.warn(
                    "pack_multipole_moments: quadrupoles have a non-zero trace "
                    f"(max |Tr Q| = {tr_max:.3e}); the l=2 channel "
                    "is traceless, so the trace is dropped.",
                    stacklevel=2,
                )
        cols.append(cartesian_quadrupole_to_e3nn(quadrupoles))  # (N, 5)
    return torch.cat(cols, dim=-1).contiguous()
