import json
from typing import List

import consts


class Tokens:
    def __init__(self, base_stoi, merges):
        self._base_stoi = base_stoi
        self._base_itos = {v: k for k, v in base_stoi.items()}
        self._merges = merges
        self._merges_inverted = {v: k for k, v in merges}
        self._doc_end_token = self._base_stoi[consts.DOC_END_TOKEN]

    def Save(self, path):
        with open(path, "w") as f:
            json.dump({"stoi": self._base_stoi, "merges": self._merges}, f)

    def Size(self):
        return len(self._base_stoi) + len(self._merges)

    # Because of MusicXML format, we know the base start token will always be
    # merged a lot with other tokens. So let's return all BPE'ed tokens that
    # begin with the start token ("score-partwise"). A way to avoid this in the
    # future would be to always keep the first base token separate.
    def GetStartsOrDie(self):
        # Find start base token.
        start_base_token = -1
        for k, v in self._base_stoi.items():
            if "score-partwise" in k:
                start_base_token = v
        assert start_base_token != -1

        # Find all BPE'ed tokens that begin with the start base token.
        res = []
        for k in self._merges_inverted.keys():
            translated = self.Translate(k)
            if translated[0] == start_base_token:
                res.append(k)
        return res

    def Translate(self, tok: int) -> List[int]:
        if tok in self._base_stoi:
            return [tok]
        res = []
        stack = [tok]
        while stack:
            tok = stack.pop()
            if tok in self._base_itos:
                res.append(tok)
            else:
                a, b = self._merges_inverted[tok]
                stack.append(b)
                stack.append(a)
        return res

    def TokenIsTag(self, tok: int) -> bool:
        return (
            self._base_itos.get(tok, "").startswith("<")
            and not tok == self._doc_end_token
        )

    def GetStr(self, tok: int) -> str:
        return self._base_itos[tok]

    def GetCloseTagStr(self, tok: int) -> str:
        assert self.TokenIsTag(tok)
        s = self.GetStr(tok)
        tag_name = s.split(" ")[0][1:]
        tag_name = tag_name if not tag_name.endswith(">") else tag_name[:-1]
        return f"</{tag_name}>"

    @staticmethod
    def Load(path) -> "Tokens":
        with open(path, "r") as f:
            data = json.load(f)
            return Tokens(data["stoi"], data["merges"])
