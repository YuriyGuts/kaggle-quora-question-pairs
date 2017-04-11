.PHONY: glove

glove:
	./utils/preprocess_stanford_glove.py data/aux/glove.6B.50d.txt     50   `cat data/aux/glove.6B.50d.txt | wc -l`
	./utils/preprocess_stanford_glove.py data/aux/glove.6B.100d.txt    100  `cat data/aux/glove.6B.100d.txt | wc -l`
	./utils/preprocess_stanford_glove.py data/aux/glove.6B.300d.txt    300  `cat data/aux/glove.6B.300d.txt | wc -l`
	./utils/preprocess_stanford_glove.py data/aux/glove.840B.300d.txt  300  `cat data/aux/glove.840B.300d.txt | wc -l`
	./utils/preprocess_stanford_glove.py data/aux/glove.42B.300d.txt   300  `cat data/aux/glove.42B.300d.txt | wc -l`

	mv glove*.dill data/aux
