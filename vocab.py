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
