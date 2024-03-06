from dataclasses import dataclass
import collections
from multiprocessing import Pool, Lock

import consts
import sbiff


lock = Lock()


@dataclass
class BpeOptions:
    # The base stoi before running BPE.
    stoi: dict
    # The path to the sbiff file to read from.
    src: str
    # The path to the sbiff file to write to.
    dst: str
    # The maximum number of tokens to process for Bpe. These numbers must fit
    # into memory. For now let's go with 2GB.
    tokens_to_process: int = 1024 * 1024 * 1024 * 2
    # The maximum desired number of tokens in the vocabulary.
    max_vocab_size = consts.MAX_NUM_DESIRED_TOKENS
    # Useful for controlling in a test.
    parallelism = consts.PARALLELISM


def _Validate(options: BpeOptions, ints: list[int]):
    itos = {i: s for s, i in options.stoi.items()}
    produced = []
    for i in ints:
        d = collections.deque([i])
        while d:
            num = d.popleft()
            if type(itos[num]) == tuple:
                d.appendleft(itos[num][1])
                d.appendleft(itos[num][0])
            else:
                produced.append(num)

    original = sbiff.ReadAllInts(options.src)
    assert produced == original, f"Produced={produced} Original={original}"


def _Merge(pair, new_int, ints):
    new_ints = []
    j = 0
    while j < len(ints):
        if j + 1 != len(ints) and ints[j] == pair[0] and ints[j + 1] == pair[1]:
            new_ints.append(new_int)
            j += 2
        else:
            new_ints.append(ints[j])
            j += 1
    return new_ints


def _GenNewTokens(options: BpeOptions):
    i = len(options.stoi)
    end_token = options.stoi[consts.DOC_END_TOKEN]

    print(
        "Running bpe. Original vocab size:",
        i,
        "Desired size:",
        options.max_vocab_size,
        "Need to make this many more tokens:",
        options.max_vocab_size - i,
    )

    merges = []

    ints = sbiff.ReadUpToNInts(options.src, n=options.tokens_to_process)
    print("First 30 ints:", ints[:30])

    # Iterate over dataset finding the pair that occurs most frequently.
    # TODO: Consider rewriting in Go/C++/etc. for speed.
    for _ in range(options.max_vocab_size - i):
        pair_counts = collections.Counter()
        most_common_pair = (-1, -1)
        most_common_count = -1
        prev = None
        prev_prev = None
        for a, b in zip(ints, ints[1:]):
            if a == end_token or b == end_token:
                # Never pair with the end token.
                continue

            if (a, b) == prev and (a, b) != prev_prev:
                # Don't count the same pair twice in a row unless it's also the
                # pair before that (because 1,1,1,1 -> 2,2 is fine).
                prev_prev = prev
                prev = (a, b)
                continue

            pair_counts[(a, b)] += 1
            if pair_counts[(a, b)] > most_common_count:
                most_common_pair = (a, b)
                most_common_count = pair_counts[(a, b)]
            prev_prev = prev
            prev = (a, b)

        # print("Most common pair:", most_common_pair, "count:", most_common_count)
        if most_common_count == 1:
            # TODO: Maybe we should stop far before this as merging tokens of
            # length 2 might be a bad idea as well. A pair only occuring twice
            # is probably very rare within a dataset.
            print("No more pairs to merge. Stopping at", i, "tokens.")
            break
        merges.append((most_common_pair, i))
        print("Creating token", "len(ints):", len(ints), most_common_pair, "->", i)

        # Replace all occurrences of the pair with the new token.
        ints = _Merge(most_common_pair, i, ints)
        # print("After 30 ints:", ints[:30])
        i += 1

    # _Validate(options, ints)
    # print("Ints", ints)
    return merges


def _WriteAlteredDoc(args):
    ints, merges, options = args

    # TODO: This is brute force. Surely there's a better way.
    for pair, new_int in merges:
        ints = _Merge(pair, new_int, ints)

    lock.acquire()
    sbiff.AppendInts(options.dst, ints)
    lock.release()


# Use the new vocab to rewrite the dataset with the new tokens.
def _WriteNewDataset(merges, options: BpeOptions):
    end_token = options.stoi[consts.DOC_END_TOKEN]

    def Gen():
        print("calling gen")
        offset = 0
        while True:
            ints, nxt = sbiff.ReadUntilInt(options.src, end_token, n_offset=offset)
            if ints:
                ints.append(end_token)
                yield (ints, merges, options)

            if nxt == -1:
                return

            offset = nxt + 1

    with Pool(processes=consts.PARALLELISM) as pool:
        for _ in pool.imap(_WriteAlteredDoc, Gen()):
            pass


# Given a stoi and path to an src sbiff file and a dst sbiff file, this function
# runs BPE-like tokenization on the src file and writes the result to the dst
# file. It returns the stoi with the new tokens added.
def RunBpe(options: BpeOptions):
    merges = _GenNewTokens(options)
    _WriteNewDataset(merges, options)
    return merges
