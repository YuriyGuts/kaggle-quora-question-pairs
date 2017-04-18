.PHONY: glove fasttext

glove:
	./utils/binarize_vector_file.py data/aux/glove.6B.50d.txt     50   `cat data/aux/glove.6B.50d.txt | wc -l`
	./utils/binarize_vector_file.py data/aux/glove.6B.100d.txt    100  `cat data/aux/glove.6B.100d.txt | wc -l`
	./utils/binarize_vector_file.py data/aux/glove.6B.300d.txt    300  `cat data/aux/glove.6B.300d.txt | wc -l`
	./utils/binarize_vector_file.py data/aux/glove.840B.300d.txt  300  `cat data/aux/glove.840B.300d.txt | wc -l`
	./utils/binarize_vector_file.py data/aux/glove.42B.300d.txt   300  `cat data/aux/glove.42B.300d.txt | wc -l`

	mv -f glove*.pickle data/aux

fasttext:
	echo `cat data/aux/quora_unfiltered.vocab | wc -l` 300 > data/aux/quora_unfiltered.vec
	fasttext print-vectors data/aux/fasttest.wiki.en.bin < data/aux/quora_unfiltered.vocab >> data/aux/quora_unfiltered.vec

	echo `cat data/aux/quora_filtered.vocab | wc -l` 300 > data/aux/quora_filtered.vec
	fasttext print-vectors data/aux/fasttest.wiki.en.bin < data/aux/quora_filtered.vocab >> data/aux/quora_filtered.vec

	./utils/binarize_vector_file.py data/aux/quora_unfiltered.vec        300  `cat data/aux/quora_unfiltered.vec | wc -l`
	./utils/binarize_vector_file.py data/aux/quora_filtered.vec          300  `cat data/aux/quora_filtered.vec | wc -l`

	mv -f quora*.pickle data/aux
