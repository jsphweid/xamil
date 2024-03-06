# XaMiL

This is an experiment in generating sheet music using a transformer model. I use
Karpathy's NanoGPT for the model. All the unique code in this repo is simply
transforming MusicXML files into a sequence of tokens.

My method of tokenizing MusicXML is in two parts:
1. Establish a minimal base set of tokens that gets rid of a lot of redundancy 
inherent in an XML file (ex. end tags, which can be generated at inference time).
2. Use a BPE-like algorithm to create additional tokens by merging in the most
common pairs of tokens together.

My initial attempt (2023) at this was just Part 1. It yieled a few hundred base
tokens. In early 2024, I added Part 2 which allowed me to raise the vocab to an
arbitrarily large size. In an experiment I went to 20k tokens which reduced my
overall training set by more than 10x. This also has the benefit of making my
context length effectively 10x longer and inference 10x faster.

This is still highly experimental and I didn't do a thorough analysis.


## Use
Find good XML files (single staff for now) to train on.
```bash
python find_good_files.py --xml_folder_root=/path/to/xmls
```

Prep data by creating vocab, extracting tokens, and saving files for validation.
```bash
python prep.py
```

Train the model as long as you want.
```bash
python model_train.py
```

Run inference.
```bash
python model_infer.py
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

MIT License; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
