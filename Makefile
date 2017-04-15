.PHONY: glove fasttext

glove:
	./utils/binarize_vector_file.py data/aux/glove.6B.50d.txt     50   `cat data/aux/glove.6B.50d.txt | wc -l`
	./utils/binarize_vector_file.py data/aux/glove.6B.100d.txt    100  `cat data/aux/glove.6B.100d.txt | wc -l`
	./utils/binarize_vector_file.py data/aux/glove.6B.300d.txt    300  `cat data/aux/glove.6B.300d.txt | wc -l`
	./utils/binarize_vector_file.py data/aux/glove.840B.300d.txt  300  `cat data/aux/glove.840B.300d.txt | wc -l`
	./utils/binarize_vector_file.py data/aux/glove.42B.300d.txt   300  `cat data/aux/glove.42B.300d.txt | wc -l`

	mv glove*.pickle data/aux

fasttext:
	fasttext print-vectors data/aux/fasttest.wiki.en.bin < data/preproc/quora_all.vocab > data/aux/quora_all.vec
	./utils/binarize_vector_file.py data/aux/quora_all.vec        300  `cat data/aux/quora_all.vec | wc -l`

	mv quora*.pickle data/aux
