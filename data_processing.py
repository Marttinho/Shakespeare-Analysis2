import numpy as np
import pandas as pd
import nltk
import more_itertools
import textacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

f = open('pg100.txt', mode='r', encoding='utf-8-sig')
lines = f.read()

from spacy.lang.en import English
nlp = English()
tokenizer1 = RegexpTokenizer(r'[a-z]\w+')
tokenizer = Tokenizer(nlp.vocab)

# nltk_tokens = word_tokenize(lines)
nltk_tokens = tokenizer1.tokenize(lines.lower())
bigrams = list(nltk.bigrams(nltk_tokens))
trigrams = list(nltk.trigrams(nltk_tokens))
fourgrams = list(nltk.ngrams(nltk_tokens,4))
fivegrams = list(nltk.ngrams(nltk_tokens,5))
# print(list(nltk.trigrams(nltk_tokens)))
bi_frequency = nltk.FreqDist(bigrams).most_common(10)
tri_frequency = nltk.FreqDist(trigrams).most_common(10)
four_frequency = nltk.FreqDist(fourgrams).most_common(10)
five_frequency = nltk.FreqDist(fivegrams).most_common(10)
print("Ten most common bigrams are:")
for key,value in bi_frequency:
    print(key,value)
print("Ten most common triigrams are:")
for key,value in tri_frequency:
    print(key,value)
print("Ten most common fourgrams are:")
for key,value in four_frequency:
    print(key,value)
print("Ten most common fivegrams are:")
for key,value in five_frequency:
    print(key,value)



