Parallel tokenization tool. Splits the question set into smaller batches, tokenizes them, and recombines the batches afterwards. Useful for processing a large set of questions (e.g. `test.csv`).


## Usage

Please three auxiliary files here:

* `stopwords.vocab`: list of words to be dropped during tokenization.
* `spelling_corrections.json`: spelling correction dictionary.

Adjust `INPUT_FILE`, `OUTPUT_FILE`, `LINES_PER_CHUNK` in `tokenize_test_set.sh` as needed. Run the script afterwards.
