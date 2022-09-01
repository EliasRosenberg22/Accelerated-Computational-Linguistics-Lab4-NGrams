#Author: Elias Rosenberg
#Date: April 26th, 2021
#Purpose: Train a spell-checking algorithm to correct maori instead of english.
#Input: maori training text + mispelled words
#output: correct spelling suggestion of mispelled words


import re
from collections import Counter
from nltk import word_tokenize

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('/Users/eliasrosenberg/PycharmProjects/CS72 Lab 4/cim-sentences.txt').read())) #changning the text to maori
print("Maori downloaded...")

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


#the method that takes in user input, runs the above methods to correct the sentence, and returns spelling corrections
def runProgam():
    global input

    testDict = (known(WORDS)) #list of words to compare input to properly spelled Maori words
    #print(testDict)
    print("Please enter a mispelled sentence in Cook Islands Maori and press ENTER to receive spelling correction suggestions.")
    input = input() #taking in user input

    input = input.lower()
    input = re.sub(r'[^\w\s]','', string=input) #changing input to lowercase and removing punctuation
    inputWords = word_tokenize(input)

    corrections = []
    mispellings = []
    for word in inputWords:
        if word not in testDict:
            mispellings.append(word)
            corrections.append(candidates(word))

    for i in range(len(corrections)):
        print("---possible mispelling: " + "'" + mispellings[i] + "'" + "---")
        print(corrections[i])

if __name__ == '__main__':
    runProgam()
