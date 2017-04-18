#!/usr/bin/env python3.6

import json
import pickle
import sys
import time

import nltk

import pandas as pd


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load_json(filename, **kwargs):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f, **kwargs)


def save_json(obj, filename, **kwargs):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, **kwargs)


def load_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f.readlines()]


def save_lines(lines, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def translate(text, translation):
    for token, replacement in translation.items():
        text = text.replace(token, ' ' + replacement + ' ')
    text = text.replace('  ', ' ')
    return text


def spell_digits(text):
    translation = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'nine',
        '9': 'ten',
    }
    return translate(text, translation)


def expand_negations(text):
    translation = {
        "can't": 'can not',
        "won't": 'would not',
        "shan't": 'shall not',
    }
    text = translate(text, translation)
    return text.replace("n't", " not")


def correct_spelling(text):
    return ' '.join(
        spelling_corrections.get(token, token)
        for token in tokenizer.tokenize(text)
    )


def get_question_tokens(question):
    question = question.lower()
    question = spell_digits(question)
    question = expand_negations(question)
    question = correct_spelling(question)
    
    tokens = [
        token
        for token in tokenizer.tokenize(question.lower())
        if token not in stopwords
    ]
    tokens.append('.')
    return tokens


input_filename, output_filename = sys.argv[1], sys.argv[2]

df_questions = pd.read_csv(input_filename).fillna('')
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopwords = set(load_lines('stopwords.vocab'))
spelling_corrections = load_json('spelling_corrections.json')

tokenized_questions = []

for index, row in df_questions.iterrows():
    tokenized_questions.append({
        'id': row.test_id if 'test_id' in row else row.id,
        'question1': get_question_tokens(row.question1),
        'question2': get_question_tokens(row.question2),
    })

    # Checkpoint intermediate results.    
    if index % 20000 == 0:
        save_json(tokenized_questions, output_filename)

save_json(tokenized_questions, output_filename)
