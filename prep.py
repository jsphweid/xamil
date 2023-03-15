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

from multiprocessing import Pool, Lock
import json
import collections
import struct
from typing import List, Set

import util
import consts
from vocab import Vocab


def get_unique_tokens(paths: List[str]) -> Set[str]:
    lock = Lock()
    all_unique_tokens = set()

    def cb(unique_tokens: Set[str]):
        nonlocal all_unique_tokens
        lock.acquire()
        all_unique_tokens |= unique_tokens
        lock.release()

    apply_results = []
    with Pool(consts.PARALLELISM) as p:
        for path in paths:
            apply_result = p.apply_async(util.get_unique_tokens, (path,), callback=cb)
            apply_results.append(apply_result)
        for apply_result in apply_results:
            apply_result.wait()
    return all_unique_tokens


def collect(path, stoi, tag_stoi):
    tokens = util.get_tokens_from_xml(path)
    ints = [stoi[t] for t in tokens]
    tags = [tag_stoi[t] for t in tokens if t in tag_stoi]

    # The number of tokens should always be below what can be stored in a 16-bit
    # integer. Storing them as varints would be even more space efficient but for
    # now it's completely unnecessary.
    packed = struct.pack(">" + "H" * len(ints), *ints)

    tok2tok, tag2tag = collections.defaultdict(set), collections.defaultdict(set)
    for i in range(1, len(ints)):
        tok2tok[ints[i - 1]].add(ints[i])
    for i in range(1, len(tags)):
        tag2tag[tags[i - 1]].add(tags[i])

    return packed, tok2tok, tag2tag


def sink_nums(paths: List[str], stoi: dict, tag_stoi: dict):
    """Writes tokens to file while accumulating validation dicts."""

    tok2tok, tag2tag = collections.defaultdict(set), collections.defaultdict(set)
    lock = Lock()

    def cb(args):
        nonlocal tok2tok, tag2tag
        packed, _tok2tok, _tag2tag = args
        lock.acquire()
        for k, v in _tok2tok.items():
            tok2tok[k] |= v
        for k, v in _tag2tag.items():
            tag2tag[k] |= v
        with open(consts.TRAINING_DATA_NUMS, "ab") as f:
            f.write(packed)
        lock.release()

    apply_results = []
    with Pool(consts.PARALLELISM) as p:
        for path in paths:
            apply_result = p.apply_async(collect, (path, stoi, tag_stoi), callback=cb)
            apply_results.append(apply_result)
        for apply_result in apply_results:
            apply_result.wait()

    return tok2tok, tag2tag


def run():
    # Clear any previous prepared data.
    util.clear_if_exists(consts.TRAINING_DATA_ROOT)

    # Get the paths to all the training files.
    with open(consts.TRAINING_FILES_LIST, "r") as f:
        paths = f.read().split("\n")[:-1]

    # Establish the vocabulary.
    unique_tokens = get_unique_tokens(paths)
    stoi = {token: i for i, token in enumerate(unique_tokens)}
    tag_stoi = {s: i for s, i in stoi.items() if Vocab.is_tag(s)}

    # Use vocab to write the training data to file and collect the validation dicts.
    tok2tok, tag2tag = sink_nums(paths, stoi, tag_stoi)
    tok2tok = {k: list(v) for k, v in tok2tok.items()}
    tag2tag = {k: list(v) for k, v in tag2tag.items()}

    # Dump everything else.
    with open(consts.TRAINING_DATA_ROOT + "/stoi.json", "w") as f:
        json.dump(stoi, f)
    with open(consts.TRAINING_DATA_ROOT + "/tok2tok.json", "w") as f:
        json.dump(tok2tok, f)
    with open(consts.TRAINING_DATA_ROOT + "/tag2tag.json", "w") as f:
        json.dump(tag2tag, f)


if __name__ == "__main__":
    run()
