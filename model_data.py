# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import json
import struct
import torch
from enum import Enum
import torch

import util
import consts
from vocab import Vocab


class Split(Enum):
    Train = "Train"
    Val = "Val"


def _read_ints(data, bytes_start, num_bytes):
    sl = data[bytes_start : bytes_start + num_bytes]
    return list(struct.unpack(">" + "H" * (len(sl) // struct.calcsize("H")), sl))


class ModelDataProvider:
    """Provides batches of data for training and validation."""

    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(consts.TRAINING_DATA_NUMS, "rb") as f:
            self.byte_tokens = f.read()
        split = int(0.9 * len(self.byte_tokens))
        split = split - (split & 1)  # Make sure it splits on an even number.
        self._train_bytes_segments = self.byte_tokens[:split]
        self._val_bytes_segments = self.byte_tokens[split:]

    def get_batch(self, split: Split, block_size: int, batch_size: int):
        data = (
            self._train_bytes_segments
            if split == Split.Train
            else self._val_bytes_segments
        )

        # TODO: This is unnecessarily complicated. Consider refactoring eventually.
        # Keep in mind we're reading a byte array here and each number is 2 bytes.
        # "// 2" to correctly subtract with block_size isn't defined in bytes.
        # "* 2" to adjust back into byte indices.
        # "- 2" to offset by one number (2 bytes) to associate with the Y properly.
        ix = (
            torch.randint(low=1, high=(len(data) // 2) - block_size, size=(batch_size,))
            * 2
        ) - 2
        x = torch.stack([torch.tensor(_read_ints(data, i, block_size * 2)) for i in ix])
        y = torch.stack(
            [torch.tensor(_read_ints(data, i + 2, block_size * 2)) for i in ix]
        )

        if self._device == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously non_blocking=True)
            x, y = x.pin_memory().to(
                self._device, non_blocking=True
            ), y.pin_memory().to(self._device, non_blocking=True)
        else:
            x, y = x.to(self._device), y.to(self._device)
        return x, y
