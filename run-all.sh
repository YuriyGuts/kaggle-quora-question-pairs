#!/usr/bin/env bash

run_notebook() {
    printf '%75s\n' | tr ' ' -
    echo "[$(date)] Running $1"
    jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --output output/$1 --execute $1
}

cd notebooks
mkdir -p output
rm -f output/*.nbconvert.ipynb

# Preprocessing
run_notebook preproc-tokenize-spellcheck.ipynb
run_notebook preproc-extract-unique-questions.ipynb
run_notebook preproc-embeddings-fasttext.ipynb
run_notebook preproc-nn-sequences-fasttext.ipynb

# Feature extraction
run_notebook feature-fuzzy.ipynb
run_notebook feature-jaccard-ngrams.ipynb
run_notebook feature-lda.ipynb
run_notebook feature-magic-cooccurrence-matrix.ipynb
run_notebook feature-magic-frequencies.ipynb
run_notebook feature-magic-pagerank.ipynb
run_notebook feature-nlp-tags.ipynb
run_notebook feature-phrase-embedding.ipynb
run_notebook feature-simple-summaries.ipynb
run_notebook feature-tfidf.ipynb
run_notebook feature-wmd.ipynb
run_notebook feature-wm-intersect.ipynb
run_notebook feature-wordnet-similarity.ipynb

# Feature extraction: neural networks
run_notebook feature-oofp-nn-bi-lstm-with-magic.ipynb
run_notebook feature-oofp-nn-cnn-with-magic.ipynb
run_notebook feature-oofp-nn-mlp-with-magic.ipynb
run_notebook feature-oofp-nn-siamese-lstm-attention.ipynb

# Feature extraction: 3rd party
run_notebook feature-3rdparty-abhishek.ipynb
run_notebook feature-3rdparty-dasolmar-whq.ipynb
run_notebook feature-3rdparty-image-similarity.ipynb
run_notebook feature-3rdparty-mephistopheies.ipynb

# Prediction
run_notebook classify-lightgbm-cv-pred.ipynb
