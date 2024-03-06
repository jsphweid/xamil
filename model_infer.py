# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import torch
import random

import consts
from validator import Validator
from interpreter import Interpreter
from tokens import Tokens
from model_def import get_model_and_config


tokens = Tokens.Load(os.path.join(consts.TRAINING_DATA_ROOT, "tokens.json"))


starts = tokens.GetStartsOrDie()
start = random.choice(starts)
validator = Validator(start, tokens)
interpreter = Interpreter(start, tokens)
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
