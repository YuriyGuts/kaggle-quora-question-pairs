#!/usr/bin/env python3.6

import os
import sys

import pickle
import numpy as np


def read_text_vectors(filename, dimension, count):
    vocab = {}
    vectors = np.zeros((count, dimension), dtype='float32')

    with open(filename, encoding='utf-8') as glove_file:
        for i in range(count):
            components = glove_file.readline().split()
            word = ' '.join(components[:-dimension])
            vocab[word] = i
            vectors[i, :] = np.array(components[-dimension:], dtype='float32')

    return vectors, vocab


def main():
    if len(sys.argv) < 4:
        print('Usage: binarize_vector_file.py <txtfile> <dimensions> <wordcount>')
        sys.exit(1)

    filename, dims, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    basename = os.path.splitext(os.path.basename(filename))[0]
    vectors, vocab = read_text_vectors(filename, dims, count)

    with open(basename + '.vectors.pickle', 'wb') as f:
        pickle.dump(vectors, f, protocol=4)
    with open(basename + '.vocab.pickle', 'wb') as f:
        pickle.dump(vocab, f, protocol=4)


if __name__ == '__main__':
    main()
