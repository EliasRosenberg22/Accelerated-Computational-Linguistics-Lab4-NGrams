#Author: Elias Rosenberg
#Date: April 26th, 2021
#Purpose: Take in a corpus and model an n-gram to output stretches of 100-word text based on unigram, bigram, and trigram models. Can also calculate unigram probabilities and counts.
#Input: corpus, desired n-gram number
#output: sentences in corpus language + probability & counts stats. 


# =======================================================================
# Dartmouth College, LING48, Spring 2021
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Examples for Exercise 4.1: N-gram probabilities and n-gram text generation
#
# You must study the links below and attempt to modify
# the program according to the homework instructions.
#
# Documentation of the NLTK.LM package
# https://www.nltk.org/api/nltk.lm.html
#
# How to extract n-gram probabilities
# https://stackoverflow.com/questions/54962539/how-to-get-the-probability-of-bigrams-in-a-text-of-sentences
# =======================================================================
import sys
import re
import os
import io
import random
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE, NgramCounter, Vocabulary
from nltk.util import ngrams
from collections import Counter
from nltk import word_tokenize, sent_tokenize, bigrams, trigrams

global Model

# Open file
file = io.open('/Users/eliasrosenberg/PycharmProjects/CS72 Lab 4/shakespeare.txt', encoding='utf8') #downloading the desired text to build the model
text = file.read()
print("Text file downloaded...")
print("Generating a model. It takes a while...")
strText = ""
for line in text: #just to see the text.
    strText += line

# Preprocess the tokenized text for language modelling
# https://stackoverflow.com/questions/54959340/nltk-language-modeling-confusion


def compileRegEx(regExInput): #regex to (attempt to) reel in spacing.
    return re.compile(regExInput, re.IGNORECASE)


#method that takes a number(n) to represent the desired unigram
def generateText(n):
    global Model

    type = "" #tagging the output model
    if n == 1:
        type = "Unigram"
    if n == 2:
        type = "Bigram"
    if n == 3:
        type = "Trigram"
    if n == 4:
        type = "Four-gram"


    punc = compileRegEx("\W+")

    paddedLine = [list(pad_both_ends(word_tokenize(text.lower()), n))]
    train, vocab = padded_everygram_pipeline(n, paddedLine)

    # Train an n-gram maximum likelihood estimation model.

    Model = MLE(n)
    Model.fit(train, vocab)
    print("---Generating Text : " + type + "---")
    output1 = Model.generate(num_words=100) #we want an output of 100 words
    #print(output1)


    sentence = "" #turning the list of tokens into a readable sentence.
    for words in output1:
        sentence += words
        if words != punc: #trying to make the spacing correct (Rolando says this is okay...)
            sentence += " "
    print(sentence)
    print("\n")

#takes three strings, and calculates the probability and counts of those strings
def counts_and_probabilities(string1, string2, string3):

    print("generating a new model to test trigram probabilities...")
    print("\n")
    n= 3 #generating another model

    paddedLine = [list(pad_both_ends(word_tokenize(text.lower()), n))]
    train, vocab = padded_everygram_pipeline(n, paddedLine)

    # Train an n-gram maximum likelihood estimation model.

    model = MLE(n)
    model.fit(train, vocab)

    gramCount1 = model.counts[string1]  # i.e. Count('in')
    gramCount2 = model.counts[[string1]][string2]  # i.e. Count('in fair')
    gramCount3 = model.counts[[string1, string2]][string3]  # i.e. Count('in fair verona')



    gramProb1 = model.score(".",'in'.split()) # P('in')
    gramProb2 = model.score('fair', 'in'.split())  # P('in'|'fair')
    gramProb3 = model.score('verona', 'in fair'.split())  # P('verona'|'in fair')

    print("---Counts---")
    print(string1 + ":" + str(gramCount1))
    print(string1 + " " + string2 + ":" + str(gramCount2))
    print(string1 + " " + string2 + " " + string3 + ":" + str(gramCount3))

    print("---Probabilities---")
    print(string1 + ":" + str(gramProb1))
    print(string1 + " " + string2 + ":" + str(gramProb2))
    print(string1 + " " + string2 + " " + string3 + ":" + str(gramProb3))


if __name__ == '__main__':
    generateText(1)
    generateText(2)
    generateText(3)
    generateText(4)

    counts_and_probabilities("in", "fair", "verona")