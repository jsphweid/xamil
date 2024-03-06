# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from enum import Enum
import struct
import torch
from enum import Enum
import torch
import sbiff

import consts


class Split(Enum):
    Train = "Train"
    Val = "Val"


class ModelDataProvider:
    """Provides batches of data for training and validation."""

    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_ints = sbiff.ReadAllInts(consts.TRAINING_DATA_BPE_NUMS)
        split = int(0.9 * len(self.all_ints))
        split = split - (split & 1)  # Make sure it splits on an even number.
        self._train_ints = self.all_ints[:split]
        self._val_ints = self.all_ints[split:]

    def get_batch(self, split: Split, block_size: int, batch_size: int):
        data = self._train_ints if split == Split.Train else self._val_ints
        ix = torch.randint(low=1, high=len(data) - block_size, size=(batch_size,))
        x = torch.stack([torch.tensor(data[i : i + block_size]) for i in ix])
        y = torch.stack([torch.tensor(data[i + 1 : i + block_size + 1]) for i in ix])

        if self._device == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously non_blocking=True)
            x, y = x.pin_memory().to(
                self._device, non_blocking=True
            ), y.pin_memory().to(self._device, non_blocking=True)
        else:
            x, y = x.to(self._device), y.to(self._device)
        return x, y
