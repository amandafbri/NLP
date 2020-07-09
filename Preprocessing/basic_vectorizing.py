from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
sequences_nltk = [[words_dict[word] for word in sentence] for sentence in tokens_list]
print(sequences_nltk)

# But there is this way using Tensorflow
tokenizer = Tokenizer(num_words=100, filters='', lower=False, oov_token='<OOV>')
include_punct = '!'
sentences = [re.sub(r'(['+include_punct+'])', r' \1', text) for text in sentences]
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences_tf = tokenizer.texts_to_sequences(sentences)
print(sequences_tf)

'''
 But they behave different!
 {'Hello': 1, 'world': 2, '!': 7, 'NLP': 4, 'is': 5, 'awesome': 6}
 {'!': 1, 'Hello': 2, 'world': 3, 'NLP': 4, 'is': 5, 'awesome': 6}

 The good news is that Tokenizer from Tensorflow filters a lot of stuff.
 With NLTK I needed to create additional functions to handle basic cleaning.

 In case of words outside the learned vocabulary, NLTK is going to raise KeyError.
 Tensorflow will ignore the word without any warnings, but we can add the 'oov_token' parameter to deal with that.
'''

'''
 maxlen will be the biggest sentence
 if we want to change that (maxlen=5), it's good to set the truncating part (truncating='post')
 it will change from [[3, 4, 2], [5, 6, 7, 2]] to [[3 4 2 0][5 6 7 2]]
 shape of padded will be (len(sentences), maxlen)
 '''
padded = pad_sequences(sequences_tf, padding='post')
print(padded)

'''
Useful functions
'''
vocabulary_size = len(tokenizer.word_index)  # +1 to considered the <OOV>
max_sentence_lenght = max([len(x) for x in sentences])
