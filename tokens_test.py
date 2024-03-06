# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import unittest
import util
import consts
from tokens import Tokens


class TestTokens(unittest.TestCase):

    def test_save_load(self):
        base_stoi = {"a": 0, "b": 1, consts.DOC_END_TOKEN: 2}
        merges = [((0, 1), 3)]

        tokens = Tokens(base_stoi, merges)

        with util.GetTempDir() as dir_path:
            path = os.path.join(dir_path, "tokens.json")
            tokens.Save(path)
            loaded = Tokens.Load(path)
            assert loaded.Size() == 4

    def test_finds_start(self):
        base_stoi = {"a": 0, "b": 1, "score-partwise": 2, consts.DOC_END_TOKEN: 3}
        merges = [((0, 1), 4)]

        tokens = Tokens(base_stoi, merges)
        assert tokens.GetStartOrDie() == 2

    def test_translate(self):
        base_stoi = {"a": 0, "b": 1, "c": 2, consts.DOC_END_TOKEN: 3}
        merges = [((0, 1), 4), ((4, 3), 5), ((5, 1), 6)]
        tokens = Tokens(base_stoi, merges)
        assert tokens.Translate(0) == [0]
        assert tokens.Translate(4) == [0, 1]
        assert tokens.Translate(5) == [0, 1, 3]
        assert tokens.Translate(6) == [0, 1, 3, 1]

    def test_token_is_tag(self):
        base_stoi = {
            "<hey>": 0,
            "not": 1,
            "<other>": 2,
            "score-partwise": 3,
            consts.DOC_END_TOKEN: 4,
        }
        merges = [((0, 1), 5)]
        tokens = Tokens(base_stoi, merges)
        assert tokens.TokenIsTag(0)
        assert not tokens.TokenIsTag(1)
        assert tokens.TokenIsTag(2)
        assert not tokens.TokenIsTag(3)
        assert not tokens.TokenIsTag(4)
        assert not tokens.TokenIsTag(5)


if __name__ == "__main__":
    unittest.main()
