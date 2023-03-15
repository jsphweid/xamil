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

import json

import consts


class Vocab:
    """Helper class for loading and parsing the model's vocab.

    This class can only be instantiated after `stoi.json` exists.

    A "tag" is a token that starts with "<" (and ends with ">").
    In my definition, a tag may contain attributes.
        ex. `<note>` or `<time symbol="common">`
    """

    def __init__(self) -> None:
        with open(f"{consts.TRAINING_DATA_ROOT}/stoi.json", "r") as f:
            self.stoi = json.loads(f.read())

        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.i_to_close_tag = {}

        for tok, i in self.stoi.items():
            if Vocab.is_tag(tok):
                tag_name = tok.split(" ")[0][1:]
                tag_name = tag_name if not tag_name.endswith(">") else tag_name[:-1]
                self.i_to_close_tag[i] = f"</{tag_name}>"

    def i_is_tag(self, i: int) -> bool:
        return Vocab.is_tag(self.itos[i])

    @staticmethod
    def is_tag(s: str) -> bool:
        return s.startswith("<")
