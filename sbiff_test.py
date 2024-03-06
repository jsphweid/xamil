# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import struct
import unittest
import util

from sbiff import (
    AppendInts,
    ReadAllInts,
    ReadNInts,
    ReadRandomNInts,
    ReadUpToNInts,
    ReadUntilInt,
    CountInts,
)


class TestSbiff(unittest.TestCase):

    def test_writes_correct_number_of_bytes(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            assert os.path.getsize(file_path) == 10

            AppendInts(file_path, [1, 2, 3])
            assert os.path.getsize(file_path) == 16

    def test_fails_when_larger_than_16_bits(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            with self.assertRaises(struct.error):
                AppendInts(file_path, [2**16])

    def test_reads_correct_values(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            assert ReadAllInts(file_path) == [1, 2, 3, 4, 5]

            AppendInts(file_path, [1, 2, 3])
            assert ReadAllInts(file_path) == [1, 2, 3, 4, 5, 1, 2, 3]

    def test_reads_n_ints(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            assert ReadNInts(file_path, 2) == [1, 2]
            assert ReadNInts(file_path, 2, n_offset=3) == [4, 5]

    def test_fails_reading_n_ints_when_past_end(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            with self.assertRaises(ValueError):
                ReadNInts(file_path, 3, n_offset=3)

    def test_reads_random_n_ints(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            assert ReadRandomNInts(file_path, 2) in [[1, 2], [2, 3], [3, 4], [4, 5]]

    def test_reads_random_with_start_end(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            ints = ReadRandomNInts(file_path, 2, end=2)
            assert set(ints) == {1, 2}

    def test_reads_up_to_n_ints(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            assert ReadUpToNInts(file_path, 2) == [1, 2]
            assert ReadUpToNInts(file_path, 5, n_offset=1) == [2, 3, 4, 5]

    def test_read_until_int(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            assert ReadUntilInt(file_path, 3) == ([1, 2], 2)

    def test_read_until_int_with_offset(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 7, 3, 4, 5, 6, 7, 8])
            assert ReadUntilInt(file_path, 7, n_offset=3) == ([4, 5, 6], 6)

    def test_read_until_int_with_large(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            arr = [1] * 100000 + [2, 1, 1]
            AppendInts(file_path, arr)
            expected = [1] * (100000 - 5)
            assert ReadUntilInt(file_path, 2, n_offset=5) == (expected, 100000)

    def test_read_until_int_when_not_exist(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            assert ReadUntilInt(file_path, 6) == ([1, 2, 3, 4, 5], -1)

    def test_counts_num_ints_in_file(self):
        with util.GetTempDir() as dir_path:
            file_path = os.path.join(dir_path, "test.bin")
            AppendInts(file_path, [1, 2, 3, 4, 5])
            assert CountInts(file_path) == 5


if __name__ == "__main__":
    unittest.main()
