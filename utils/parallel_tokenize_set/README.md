Parallel tokenization tool. Splits the question set into smaller batches, tokenizes them, and recombines the batches afterwards. Useful for processing a large set of questions (e.g. `test.csv`).


## Usage

Please two auxiliary files here:

* `custom_valid_words.vocab`: list of words to be considered valid by the spellchecker. This could be a vocabulary from the pretrained word vectors.
* `stopwords.vocab`: list of words to be dropped during tokenization.

Adjust `INPUT_FILE`, `OUTPUT_FILE`, `LINES_PER_CHUNK` in `tokenize_test_set.sh` as needed. Run the script afterwards.
