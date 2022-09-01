#Author: Elias Rosenberg
#Date: April 26th, 2021
#Purpose: Take in 3 sentences and return their perplexity
#Input: test sentences
#output: perplexity stats


# =======================================================================
# Dartmouth College, LING48, Spring 2021
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Examples for Homework 4.1: N-gram probabilities and n-gram text generation
#
# You must study the links below and attempt to modify
# the program according to the homework instructions.
#
# Documentation of the NLTK.LM package
# https://www.nltk.org/api/nltk.lm.html
#
# How to extract n-gram probabilities
# https://stackoverflow.com/questions/54962539/how-to-get-the-probability-of-bigrams-in-a-text-of-sentences
#
# Calculating perplexity with NLTK
# https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk
# =======================================================================

import os
import io
import random
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE, NgramCounter, Vocabulary
from nltk.util import ngrams
from collections import Counter
from nltk import word_tokenize, sent_tokenize, bigrams, trigrams

# Open file
file = io.open('/Users/eliasrosenberg/PycharmProjects/CS72 Lab 4/shakespeare.txt', encoding='utf8')
text = file.read()

# Preprocess the tokenized text for language modelling
# https://stackoverflow.com/questions/54959340/nltk-language-modeling-confusion
n = 2
paddedLine = [list(pad_both_ends(word_tokenize(text.lower()), n))]
train, vocab = padded_everygram_pipeline(n, paddedLine)

# Lets train a n-gram maximum likelihood estimation model.
model = MLE(n)
model.fit(train, vocab)

print(" ")

# NLTK will calculate the perplexity of these sentences
test_sentences = ['i am proud to please you', 'i like thy brother bassianus', 'i drive the car'] #changed test sentences
tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in test_sentences]

# Probability of bigrams
test_data = [bigrams(t, pad_right=False, pad_left=False) for t in tokenized_text]
for test in test_data:
    print("MLE Estimates:", [((ngram[-1], ngram[:-1]), model.score(ngram[-1], ngram[:-1])) for ngram in test])

print("")

# Perplexity of bigrams
test_data = [bigrams(t, pad_right=False, pad_left=False) for t in tokenized_text]
for i, test in enumerate(test_data):
    print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))

print("")
