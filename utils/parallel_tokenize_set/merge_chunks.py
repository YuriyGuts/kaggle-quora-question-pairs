#!/usr/bin/env python3.6

import glob
import json
import sys

from operator import itemgetter


questions = []


for filename in glob.glob('x*.json'):
    with open(filename, encoding='utf-8') as f:
        questions.extend(json.load(f))

with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump(sorted(questions, key=itemgetter('id')), f)
