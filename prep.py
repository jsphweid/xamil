# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# TODO: Refactor this because it's quite messy.

from multiprocessing import Pool, Lock
import json
import collections
import sbiff
from typing import List, Set
from absl import app, flags

import util
import consts
from vocab import Vocab
from bpe import RunBpe, BpeOptions
from tokens import Tokens


flags.DEFINE_integer(
    "max_paths",
    None,
    "The maximum number of paths to process. If None, all paths will be processed.",
)

flags.DEFINE_integer(
    "max_base_tokens_for_bpe",
    50000000,
    "The maximum number of base tokens to use for BPE.",
)


def GetUniqueTokens(paths: List[str]) -> Set[str]:
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
            apply_result = p.apply_async(util.GetUniqueTokens, (path,), callback=cb)
            apply_results.append(apply_result)
        for apply_result in apply_results:
            apply_result.wait()
    return all_unique_tokens


def Collect(path, stoi, tag_stoi):
    tokens = util.GetTokensFromXml(path)
    tokens.append(consts.DOC_END_TOKEN)
    ints = [stoi[t] for t in tokens]
    tags = [tag_stoi[t] for t in tokens if t in tag_stoi]

    tok2tok, tag2tag = collections.defaultdict(set), collections.defaultdict(set)
    for i in range(1, len(ints)):
        tok2tok[ints[i - 1]].add(ints[i])
    for i in range(1, len(tags)):
        tag2tag[tags[i - 1]].add(tags[i])

    return ints, tok2tok, tag2tag


def SinkNums(paths: List[str], stoi: dict, tag_stoi: dict):
    """Writes tokens to file while accumulating validation dicts."""

    tok2tok, tag2tag = collections.defaultdict(set), collections.defaultdict(set)
    lock = Lock()

    def cb(args):
        nonlocal tok2tok, tag2tag
        ints, _tok2tok, _tag2tag = args
        lock.acquire()
        for k, v in _tok2tok.items():
            tok2tok[k] |= v
        for k, v in _tag2tag.items():
            tag2tag[k] |= v
        sbiff.AppendInts(consts.TRAINING_DATA_NUMS, ints)
        lock.release()

    apply_results = []
    with Pool(consts.PARALLELISM) as p:
        for path in paths:
            apply_result = p.apply_async(Collect, (path, stoi, tag_stoi), callback=cb)
            apply_results.append(apply_result)
        for apply_result in apply_results:
            apply_result.wait()

    return tok2tok, tag2tag


def Main(argv):
    # Clear any previous prepared data.
    util.ClearIfExists(consts.TRAINING_DATA_ROOT)

    util.EnsureDirExists(consts.TRAINING_DATA_ROOT)

    # Get the paths to all the training files.
    with open(consts.TRAINING_FILES_LIST, "r") as f:
        paths = f.read().split("\n")[:-1]

    if flags.FLAGS.max_paths is not None:
        paths = paths[: flags.FLAGS.max_paths]

    # Establish the base vocabulary.
    unique_tokens = GetUniqueTokens(paths)
    unique_tokens.add(consts.DOC_END_TOKEN)

    stoi = {token: i for i, token in enumerate(unique_tokens)}
    tag_stoi = {s: i for s, i in stoi.items() if Vocab.is_tag(s)}

    # Use vocab to write the training data to file and collect the validation dicts.
    tok2tok, tag2tag = SinkNums(paths, stoi, tag_stoi)
    tok2tok = {k: list(v) for k, v in tok2tok.items()}
    tag2tag = {k: list(v) for k, v in tag2tag.items()}

    # Dump these now -- we'll use them to validate things later.
    with open(consts.TRAINING_DATA_ROOT + "/tok2tok.json", "w") as f:
        json.dump(tok2tok, f)
    with open(consts.TRAINING_DATA_ROOT + "/tag2tag.json", "w") as f:
        json.dump(tag2tag, f)

    merges = RunBpe(
        BpeOptions(
            stoi=stoi,
            src=consts.TRAINING_DATA_NUMS,
            dst=consts.TRAINING_DATA_BPE_NUMS,
        )
    )
    Tokens(stoi, merges).Save(consts.TRAINING_DATA_ROOT + "/tokens.json")


if __name__ == "__main__":
    app.run(Main)
