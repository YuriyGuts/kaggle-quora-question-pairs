#!/usr/bin/env python3.6

import json
import sys


input_files = sys.argv[1:]
vocab = set()

for filename in input_files:
    with open(filename, encoding='utf-8') as f:
        pairs = json.load(f)
        for pair in pairs:
            vocab.update(set(pair['question1']))
            vocab.update(set(pair['question2']))


with open('quora.vocab', 'w') as f:
    f.write('\n'.join(sorted(list(vocab))))
