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

import os
import torch

import consts
from validator import Validator
from interpreter import Interpreter
from vocab import Vocab
from model_def import get_model_and_config


vocab = Vocab()


# Find token containing score-partwise because this must be the start token.
start = None
for k, v in vocab.stoi.items():
    if "score-partwise" in k:
        start = v
        break
assert start is not None

validator = Validator(start, vocab)
interpreter = Interpreter(start, vocab)
ckpt_path = os.path.join(consts.MODEL_DATA_ROOT, "ckpt.pt")
checkpoint = torch.load(
    ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
)
state_dict = checkpoint["model"]
model, _ = get_model_and_config()
model.load_state_dict(state_dict)
model.generate(
    torch.tensor([[start]]),
    max_new_tokens=5000,
    validator=validator,
    interpreter=interpreter,
)
tokens = res[0].tolist()
