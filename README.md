# XaMiL

MusicXML with a little AI thrown in.

This is an experiment in generating sheet music using a transformer model. I use
Karpathy's NanoGPT for the model. All the unique code in this repo is simply
transforming MusicXML files into a sequence of tokens.

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

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
