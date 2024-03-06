# A file to manage reading from and writing to my makeshift "record" format.
# All tokens are stored as 2-byte integers. The record file is nothing but a
# sequence of these integers. SBIFF stands for "Sixteen Bit Integers File
# Format".

import os
import struct
from typing import List, Tuple
import random
import collections


_INT_SIZE = struct.calcsize("H")


def AppendInts(file_path: str, ints: List[int]):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "ab") as f:
        f.write(struct.pack(">" + "H" * len(ints), *ints))


def ReadAllInts(file_path: str) -> List[int]:
    with open(file_path, "rb") as f:
        b = f.read()
    return list(struct.unpack(">" + "H" * (len(b) // _INT_SIZE), b))


# NOTE: n_offset is in terms of 2-byte integers, not bytes. n_offset=3 means
# we'll start on the 3rd number (0 index) i.e. there will be 3 numbers that
# are skipped.
def ReadNInts(file_path: str, n: int, n_offset=0) -> List[int]:
    with open(file_path, "rb") as f:
        f.seek(n_offset * _INT_SIZE)
        b = f.read(n * _INT_SIZE)
    if len(b) < n * _INT_SIZE:
        raise ValueError("File does not contain enough data from the offset.")
    return list(struct.unpack(">" + "H" * n, b))


# Similar to ReadNInts but doesn't raise an error if there's not enough data.
def ReadUpToNInts(file_path: str, n: int, n_offset=0) -> List[int]:
    with open(file_path, "rb") as f:
        f.seek(n_offset * _INT_SIZE)
        b = f.read(n * _INT_SIZE)
    return list(struct.unpack(">" + "H" * (len(b) // _INT_SIZE), b))


def ReadRandomNInts(file_path: str, n: int, end=-1) -> List[int]:
    num_ints_in_file = os.path.getsize(file_path) // _INT_SIZE
    end = num_ints_in_file if end == -1 else end
    n_offset = random.randint(0, end - n)
    return ReadNInts(file_path, n, n_offset)


def ReadUntilInt(file_path: str, i: int, n_offset=0) -> Tuple[List[int], int]:
    chunk_size = 4096
    ints = []
    overall_index = n_offset
    needle = struct.pack(">" + "H", i)
    with open(file_path, "rb") as f:
        f.seek(n_offset * _INT_SIZE)
        while True:
            chunk = f.read(chunk_size)
            b_index = chunk.find(needle)
            if b_index != -1:
                chunk = chunk[:b_index]
                ints.extend(
                    list(struct.unpack(">" + "H" * (b_index // _INT_SIZE), chunk))
                )
                return ints, overall_index + (b_index // 2)

            ints.extend(struct.unpack(">" + "H" * (len(chunk) // _INT_SIZE), chunk))
            overall_index += len(chunk) // _INT_SIZE

            if len(chunk) < chunk_size:
                break
    return ints, -1


# Reads the up to `n` tokens after `n` instances of `i` and collects results
# into a counter. Useful for debugging.
def ReadIntsAfter(file_path: str, i: int, n=1000):
    counter = collections.Counter()
    offset = 0
    for _ in range(n):
        _, i = ReadUntilInt(file_path, i, n_offset=offset)
        print(i)
        break


# Count the number of 2-byte integers in the file.
def CountInts(file_path: str) -> int:
    with open(file_path, "rb") as f:
        return len(f.read()) // _INT_SIZE
