import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense
# pip install git+https://www.github.com/keras-team/keras-contrib.git
from keras_contrib.layers import CRF


# This was the first Bi-LSTM-CRF I used.
# It was taken and adapted from https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/
def bilstm_crf_model(max_length: int, n_words: int, n_tags: int):

    input = Input(shape=(max_length,))
    model = Embedding(input_dim=n_words + 1, output_dim=20, input_length=max_length)(input)
    model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(50, activation="relu"))(model)
    crf = CRF(n_tags)
    out = crf(model)

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    print(model.summary())

    return model


# This one I saw in "NLP in Tensorflow" from Coursera
def bilstm_model(vocab_size: int):
    model = tf.keras.Sequential(
      [
          layers.Embedding(vocab_size, 64),
          # We could also stack two LSTM layers, using return_sequences=True in the first one
          layers.Bidirectional(layers.LSTM(64)),
          layers.Dense(64, activation='relu'),
          layers.Dense(1, activation='sigmoid')
      ]
    )

    print(model.summary())
    return model


bilstm_crf_model(146, 6112, 121)
bilstm_model(6112)
