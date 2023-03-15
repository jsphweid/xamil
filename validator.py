import json

import consts
from vocab import Vocab


class Validator:
    """Class used in model inference to validate the model's output.

    It's hard for the model to generate perfect valid MusicXML. This class
    provides a mechanism to validate the output of the model, raising an
    exception if the output is "invalid". The model can use this behavior
    to simply regenerate a new token until it that token is valid.

    Since our training data is known to contain correct XML, we can store
    information about the training data to use as a reference for validation.
    We do this in the form of bigram-like dictionaries, where the keys are
    tokens and the values are sets of tokens that can follow that token.

    In addition to validating token-to-token transitions, we also validate
    tag-to-tag transitions. This is because proper MusicXML requires that
    tags be properly nested.
    """

    def __init__(self, start_token: int, vocab: Vocab):
        with open(f"{consts.TRAINING_DATA_ROOT}/tok2tok.json", "r") as f:
            tok2tok_data = json.loads(f.read())
        with open(f"{consts.TRAINING_DATA_ROOT}/tag2tag.json", "r") as f:
            tag2tag_data = json.loads(f.read())
        self.tok2tok_lookup = {int(k): set(v) for k, v in tok2tok_data.items()}
        self.tag2tag_lookup = {int(k): set(v) for k, v in tag2tag_data.items()}
        self.last_token = start_token
        self.last_tag = start_token
        self.vocab = vocab

    def register_new_token(self, tok: str):
        """Saves the token presuming it is valid.

        Raises:
            Exception: If the token is not valid.
        """
        if tok not in self.tok2tok_lookup[self.last_token]:
            raise Exception()
        if self.vocab.i_is_tag(tok) and tok not in self.tag2tag_lookup[self.last_tag]:
            raise Exception()

        self.last_token = tok
        if self.vocab.i_is_tag(tok):
            self.last_tag = tok
