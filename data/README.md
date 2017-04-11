# Data Preparation


## Kaggle Datasets

Place `test.csv` and `train.csv` into the `data` folder.


## GloVe Vectors

Download the following files from [Stanford GloVe page](https://nlp.stanford.edu/projects/glove/) and unpack them to `data/aux`:

* `glove.6B.zip`
* `glove.42B.300d.zip`
* `glove.840B.300d.zip`

Run `make glove`. Preprocessed GloVe vectors and vocabularies will be saved to `data/aux/*.dill`.
