# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json

import consts
from tokens import Tokens


class Validator:
    """Class used in model inference to validate the model's output.

    It's hard for the model to generate perfect valid MusicXML. This class
    provides a mechanism to validate the output of the model, raising an
    exception if the output is "invalid". The model can use this behavior
    to simply regenerate a new token until that token is valid.

    Since our training data is known to contain correct XML, we can store
    information about the training data to use as a reference for validation.
    We do this in the form of bigram-like dictionaries, where the keys are
    tokens and the values are sets of tokens that can follow that token.

    In addition to validating token-to-token transitions, we also validate
    tag-to-tag transitions. This is because proper MusicXML requires that
    tags be properly nested.

    This is all complicated by the fact that our BPE-like tokenization adds
    tokens on the base set of tokens directly from MusicXML. Basically we have
    to take a guess, expand it into base tokens, and validate that first.
    """

    def __init__(self, start_token: int, tokens: Tokens):
        with open(f"{consts.TRAINING_DATA_ROOT}/tok2tok.json", "r") as f:
            tok2tok_data = json.loads(f.read())
        with open(f"{consts.TRAINING_DATA_ROOT}/tag2tag.json", "r") as f:
            tag2tag_data = json.loads(f.read())
        self.tok2tok_lookup = {int(k): set(v) for k, v in tok2tok_data.items()}
        self.tag2tag_lookup = {int(k): set(v) for k, v in tag2tag_data.items()}
        self.tokens = tokens

        seq = tokens.Translate(start_token)
        if len(seq) == 1:
            self.last_token = start_token
            self.last_tag = start_token
        else:
            self.last_token = seq[-1]
            seq_tags = [t for t in seq if self.tokens.TokenIsTag(t)]
            self.last_tag = seq_tags[-1] if seq_tags else self.last_tag

    def register_new_token(self, tok: str):
        """Saves the token presuming it is valid.

        Raises:
            Exception: If the token is not valid.
        """

        seq = self.tokens.Translate(tok)
        # Check tok2tok.
        if seq[0] not in self.tok2tok_lookup[self.last_token]:
            raise Exception()

        # Check tag2tag.
        # Find first and last tags.
        seq_tags = [t for t in seq if self.tokens.TokenIsTag(t)]
        if seq_tags:
            if seq_tags[0] not in self.tag2tag_lookup[self.last_tag]:
                raise Exception()

        # If we made it past validation, update state.
        self.last_token = seq[-1]
        self.last_tag = seq_tags[-1] if seq_tags else self.last_tag
