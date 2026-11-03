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

"""DDP training with complete systems packed independently on each CUDA device."""

from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

from nvalchemiops.torch.segment_mm import segment_mm  # noqa: E402

from .test_segment_mm import _differentiable_reference  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_LENGTHS = (
    ((3, 0, 5), (0, 4, 3)),
    ((0, 0, 0), (2, 0, 7)),
    ((0, 6, 2), (0, 0, 0)),
)


class _EnergyModel(torch.nn.Module):
    """Put trainable parameters on both sides of the segmented product."""

    def __init__(self, *, bf16, forces, reference=False):
        super().__init__()
        self.feature_scale = torch.nn.Parameter(torch.linspace(0.7, 1.2, 7))
        self.expert_weights = torch.nn.Parameter(
            torch.linspace(-0.4, 0.5, 3 * 7 * 5).reshape(3, 7, 5)
        )
        self.bf16 = bf16
        self.forces = forces
        self.reference = reference

    def forward(self, positions, coefficients, lengths):
        """Compute local forces before DDP prepares its outer backward pass."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            features = torch.sin(positions) * self.feature_scale
            weights = self.expert_weights * coefficients[:, None, None]
            if self.reference:
                dtype = torch.bfloat16 if self.bf16 else torch.float32
                out = _differentiable_reference(
                    features.to(dtype), weights.to(dtype), lengths
                )
            else:
                out = segment_mm(features, weights, lengths)
            values = out.float()
            energy = (torch.sin(values) + 0.1 * values.square()).sum()
        forces = None
        if self.forces:
            forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        return out, energy, forces


def _batch(rank, step, device):
    """Use different local data without storing rank-specific model buffers."""
    positions = (
        (
            torch.sin(
                torch.arange(12 * 7).reshape(12, 7) * 0.19 + rank * 0.7 + step * 0.23
            )
            * 0.5
        )
        .to(device)
        .requires_grad_()
    )
    coefficients = torch.tensor([0.7, 1.0, 1.4], device=device) + 0.13 * rank
    target = 0.1 * torch.cos(positions.detach() + 0.3 * rank)
    return positions, coefficients, target


def _loss(outputs, target, used, forces):
    """Use fixed-capacity sums so empty local batches produce connected zeros."""
    _, energy, predicted_forces = outputs
    if forces:
        return (predicted_forces[:used] - target[:used]).square().sum()
    return energy.square()


def _ddp_worker(rank, world_size, rendezvous, bf16, forces):
    """Compare real DDP reduction with dense losses evaluated on one local GPU."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.distributed.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        model = _EnergyModel(bf16=bf16, forces=forces).to(device)
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        reference = _EnergyModel(bf16=bf16, forces=forces, reference=True).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
        lengths = torch.zeros(3, device=device, dtype=torch.int32)
        rtol, atol = (3e-2, 3e-3) if bf16 else (3e-5, 3e-5)

        for step, all_lengths in enumerate(_LENGTHS):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            reference.load_state_dict(model.state_dict())
            local_lengths = all_lengths[rank]
            lengths.copy_(torch.tensor(local_lengths, device=device, dtype=torch.int32))
            positions, coefficients, target = _batch(rank, step, device)
            actual = ddp(positions, coefficients, lengths)
            loss = _loss(actual, target, sum(local_lengths), forces)

            dense_losses = []
            local_expected = None
            for source_rank in range(world_size):
                ref_positions, ref_coefficients, ref_target = _batch(
                    source_rank, step, device
                )
                ref_lengths = all_lengths[source_rank]
                expected = reference(ref_positions, ref_coefficients, ref_lengths)
                dense_losses.append(
                    _loss(expected, ref_target, sum(ref_lengths), forces)
                )
                if source_rank == rank:
                    local_expected = expected

            # The reference averages both data sets locally; it does not use a
            # distributed reduction or the SegMM operation under test.
            (sum(dense_losses) / world_size).backward()
            loss.backward()

            for value, expected in zip(actual, local_expected, strict=True):
                if value is not None:
                    assert value.dtype == expected.dtype
                    torch.testing.assert_close(value, expected, rtol=rtol, atol=atol)
            assert actual[0].dtype == (torch.bfloat16 if bf16 else torch.float32)
            assert torch.count_nonzero(actual[0][sum(local_lengths) :]) == 0
            if forces:
                assert torch.count_nonzero(actual[2][sum(local_lengths) :]) == 0

            before = [parameter.detach().clone() for parameter in model.parameters()]
            for parameter, expected in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                assert parameter.grad is not None and expected.grad is not None
                assert parameter.grad.dtype == torch.float32
                assert torch.isfinite(parameter.grad).all()
                torch.testing.assert_close(
                    parameter.grad, expected.grad, rtol=rtol, atol=atol
                )
                if world_size == 2 and sum(local_lengths) == 0:
                    # With no local atoms, an unsynchronized gradient is zero.
                    # Require enough peer signal to fail even with BF16 tolerance.
                    assert not torch.allclose(
                        torch.zeros_like(expected.grad),
                        expected.grad,
                        rtol=rtol,
                        atol=atol,
                    )

            optimizer.step()
            reference_optimizer.step()
            for parameter, expected, previous in zip(
                model.parameters(), reference.parameters(), before, strict=True
            ):
                torch.testing.assert_close(
                    parameter.detach() - previous,
                    expected.detach() - previous,
                    rtol=rtol,
                    atol=atol * 1e-3 + 2e-7,
                )
    finally:
        torch.distributed.destroy_process_group()


class TestSegmentMMDistributed:
    """Exercise local operator autograd under actual DDP gradient reduction."""

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("world_size", [1, 2], ids=["one_gpu", "two_gpus"])
    @pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16_autocast"])
    @pytest.mark.parametrize("forces", [False, True], ids=["energy", "force_loss"])
    def test_training_gradients(self, tmp_path, world_size, bf16, forces):
        """Changing lengths and empty ranks preserve the global training gradient."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_nccl_available()
        ):
            pytest.skip("NCCL required")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Requires {world_size} CUDA devices")
        torch.multiprocessing.spawn(
            _ddp_worker,
            args=(world_size, str(tmp_path / "process_group"), bf16, forces),
            nprocs=world_size,
            join=True,
        )
