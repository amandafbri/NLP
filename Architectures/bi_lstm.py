import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, Dense

from Preprocessing.basic_vectorizing import tokenizer

model = tf.keras.Sequential(
    Embedding(tokenizer.vocab_size, 64),
    # empilhar duas LSTM fica melhor, usando return_sequences=True na primeira
    Bidirectional(tf.keras.layers.LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
)
