#!/usr/bin/env python3.6

import json
import pickle
import sys
import time

import enchant
import pylev
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


input_filename, output_filename, log_filename = sys.argv[1], sys.argv[2], sys.argv[3]


questions_train = pd.read_csv(input_filename).fillna('none')
stopwords = set(load_lines('stopwords.vocab'))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
spellchecker = enchant.DictWithPWL("en_US", 'custom_valid_words.vocab')
spellcheck_log = open(sys.argv[3], 'w')


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


def get_best_suggestion(word):
    suggestion_scores = {
        suggestion: pylev.damerau_levenshtein(word, suggestion)
        for suggestion in spellchecker.suggest(word)
    }
    if len(suggestion_scores) == 0:
        return None
    
    best_suggestion = min(suggestion_scores, key=suggestion_scores.get)
    if suggestion_scores[best_suggestion] > 4:
        return None
    
    return best_suggestion


def correct_spelling(text):
    tokens = tokenizer.tokenize(text)
    corrected_tokens = []
    
    for token in tokens:
        if not spellchecker.check(token):
            correction = get_best_suggestion(token)
            if correction:
                corrected_tokens.append(correction.lower())
                print('{} ---> {}'.format(token, correction), file=spellcheck_log, flush=True)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)
    
    return ' '.join(corrected_tokens)


def get_question_tokens(question):
    question = question.lower()
    question = spell_digits(question)
    question = expand_negations(question)
    question = correct_spelling(question)
    return [token for token in tokenizer.tokenize(question) if token not in stopwords]


tokenized_train = []


for index, row in questions_train.iterrows():
    tokenized_train.append({
        'id': row.test_id,
        'question1': get_question_tokens(row.question1),
        'question2': get_question_tokens(row.question2),
    })
    
    if index % 20000 == 0:
        save_json(tokenized_train, output_filename)


spellcheck_log.close()
save_json(tokenized_train, output_filename)
