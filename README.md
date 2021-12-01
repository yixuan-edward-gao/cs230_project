# cs230_project
Final project of CS230: Deep Learning. DL approaches to small molecule property prediction.

# Files
All source codes are located in the `code` folder.
Data files are not included. All code assumes that data files are located in a folder called `data` under the root directory.

# Usage
If data files are present, in the `code` folder, run

`python3 preprocess.py`

to process all data.

To test each model, run

`python3 [model_file].py`

for each of `smiles_onehot.py`, `fingerprint.py`, and `word_embedding.py`.

# Sources
Word embedding was generated using pretrained model at
https://github.com/samoturk/mol2vec/tree/master/examples/models

See https://github.com/samoturk/mol2vec for more details.
