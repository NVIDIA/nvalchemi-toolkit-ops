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

"""The benchmark dataset cache must not execute code from a replaced file."""

import pickle
from pathlib import Path

import pytest
import torch

systems = pytest.importorskip("benchmarks.systems")


class _TinyDataset(systems.BenchmarkDataset):
    """Two-atom dataset with a single system, for exercising the cache."""

    def __init__(self, cache_dir):
        super().__init__("tiny", cache_dir=cache_dir)

    def _get_system_ids(self):
        return ["a"]

    def _compute_system(self, system_id):
        return {"positions": torch.zeros(2, 3)}


class _Payload:
    """Pickle payload that creates a marker file if it is ever executed."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (Path(self.marker),))


def test_cache_round_trips_tensor_dicts(tmp_path):
    """A cache written by the dataset loads back with weights_only=True."""
    _TinyDataset(tmp_path)
    reloaded = _TinyDataset(tmp_path)
    assert len(reloaded) == 1
    assert reloaded[0]["positions"].shape == (2, 3)


def test_replaced_cache_file_is_not_executed(tmp_path):
    """A pickle payload swapped in for the cache is rejected, not run."""
    dataset = _TinyDataset(tmp_path)
    marker = tmp_path / "executed"
    torch.save({0: {"positions": _Payload(marker)}}, dataset._get_cache_path())
    with pytest.raises(pickle.UnpicklingError):
        _TinyDataset(tmp_path)
    assert not marker.exists()
