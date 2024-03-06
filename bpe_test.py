# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from contextlib import contextmanager
import os
import unittest
import util
import consts
import random

import sbiff
from bpe import RunBpe, BpeOptions


@contextmanager
def BpeTest():
    with util.GetTempDir() as dir_path:
        pre = os.path.join(dir_path, "pre.bin")
        post = os.path.join(dir_path, "post.bin")
        yield pre, post


class TestBpe(unittest.TestCase):

    def test_bpe_produces_correct_order(self):
        with BpeTest() as (path1, path2):
            random.seed(42)
            random_nums = [random.randint(0, 1) for _ in range(100)] + [2]
            sbiff.AppendInts(path1, random_nums)
            stoi = {"0": 0, "1": 1, consts.DOC_END_TOKEN: 2}
            stoi = RunBpe(BpeOptions(stoi=stoi, src=path1, dst=path2))
            assert sbiff.ReadAllInts(path2) == [
                14,
                9,
                6,
                0,
                8,
                1,
                7,
                4,
                13,
                3,
                10,
                6,
                4,
                11,
                1,
                10,
                3,
                6,
                7,
                3,
                11,
                13,
                7,
                1,
                14,
                8,
                2,
            ]

    def test_sequence_of_1s_0s_1(self):
        with BpeTest() as (path1, path2):
            sbiff.AppendInts(path1, [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 2])
            stoi = {"0": 0, "1": 1, consts.DOC_END_TOKEN: 2}
            stoi = RunBpe(BpeOptions(stoi=stoi, src=path1, dst=path2))
            assert sbiff.ReadAllInts(path2) == [5, 5, 1, 0, 2]

    def test_sequence_of_1s_0s_2(self):
        with BpeTest() as (path1, path2):
            sbiff.AppendInts(path1, [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2])
            stoi = {"0": 0, "1": 1, consts.DOC_END_TOKEN: 2}
            stoi = RunBpe(BpeOptions(stoi=stoi, src=path1, dst=path2))
            assert sbiff.ReadAllInts(path2) == [4, 0, 0, 0, 4, 2]

    def test_sequence_of_1s_0s_3(self):
        with BpeTest() as (path1, path2):
            sbiff.AppendInts(path1, [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2])
            stoi = {"0": 0, "1": 1, consts.DOC_END_TOKEN: 2}

            stoi = RunBpe(BpeOptions(stoi=stoi, src=path1, dst=path2))
            assert sbiff.ReadAllInts(path2) == [3, 1, 3, 3, 0, 1, 0, 2]

    def test_sequence_of_1s_0s_3(self):
        with BpeTest() as (path1, path2):
            sbiff.AppendInts(path1, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2])
            stoi = {"0": 0, "1": 1, consts.DOC_END_TOKEN: 2}

            stoi = RunBpe(BpeOptions(stoi=stoi, src=path1, dst=path2))
            assert sbiff.ReadAllInts(path2) == [4, 4, 4, 3, 2]

    def test_sequence_of_1s_0s_4(self):
        with BpeTest() as (path1, path2):
            arr = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 2]
            sbiff.AppendInts(path1, arr)
            stoi = {"0": 0, "1": 1, consts.DOC_END_TOKEN: 2}

            stoi = RunBpe(BpeOptions(stoi=stoi, src=path1, dst=path2))
            assert sbiff.ReadAllInts(path2) == [3, 4, 7, 3, 1, 7, 2]

    def test_bpe_produces_correct_order_more_tokens(self):
        with util.GetTempDir() as dir_path:
            pre = os.path.join(dir_path, "pre.bin")
            post = os.path.join(dir_path, "post.bin")
            sbiff.AppendInts(pre, [0, 1, 0, 1, 0, 1, 2, 0, 2, 0, 1, 2, 3])
            stoi = {"1": 0, "2": 1, "3": 2, consts.DOC_END_TOKEN: 3}

            RunBpe(BpeOptions(stoi=stoi, src=pre, dst=post))
            assert sbiff.ReadAllInts(post) == [4, 4, 5, 0, 2, 5, 3]

    def test_bpe_produces_2(self):
        with util.GetTempDir() as dir_path:
            pre = os.path.join(dir_path, "pre.bin")
            post = os.path.join(dir_path, "post.bin")
            sbiff.AppendInts(pre, [0, 1, 0, 1, 0, 1, 3, 0, 2, 0, 1, 2, 3])
            stoi = {"1": 0, "2": 1, "3": 2, consts.DOC_END_TOKEN: 3}

            options = BpeOptions(stoi=stoi, src=pre, dst=post)
            options.parallelism = 1
            RunBpe(options)
            assert sbiff.ReadAllInts(post) == [4, 4, 4, 3, 0, 2, 4, 2, 3]


if __name__ == "__main__":
    unittest.main()
