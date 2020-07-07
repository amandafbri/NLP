from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
import re
from nltk.tokenize import word_tokenize

nltk.download('punkt')

sentences = [
    'Hello world!',
    'NLP is awesome!'
]


# First time doing vectorization was like this
def words_to_int(words: list):
    word2int = {w: i + 1 for i, w in enumerate(words)}
    return word2int


tokens_list = [word_tokenize(sentence) for sentence in sentences]
flat_tokens_list = [item for sublist in tokens_list for item in sublist]
words_dict = words_to_int(flat_tokens_list)
print(words_dict)

# But there is this way using Tensorflow
tokenizer = Tokenizer(num_words=100, filters='', lower=False)
include_punct = '!'
sentences = [re.sub(r'(['+include_punct+'])', r' \1', text) for text in sentences]
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

'''
 But they behave different!
 {'Hello': 1, 'world': 2, '!': 7, 'NLP': 4, 'is': 5, 'awesome': 6}
 {'!': 1, 'Hello': 2, 'world': 3, 'NLP': 4, 'is': 5, 'awesome': 6}

 The good news is that Tokenizer from Tensorflow filters a lot of stuff.
 With NLTK I needed to create additional functions to handle basic cleaning.
'''
