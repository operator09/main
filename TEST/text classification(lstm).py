import os
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train = pd.read_csv('C:\\Users\\82104\\PycharmProjects\\main\\TEST\\data\\train.csv')
test = pd.read_csv('C:\\Users\\82104\\PycharmProjects\\main\\TEST\\data\\test.csv')
submission = pd.read_csv("C:\\Users\\82104\\PycharmProjects\\main\\TEST\\data\\sample_submission.csv")


labels = train.target
sentence = train.text

from sklearn.model_selection import train_test_split
train_sentence, test_sentence, train_labels, test_labels = train_test_split(sentence, labels, test_size= 0.2, random_state=2020)

vocab_size = 1000
token = Tokenizer(num_words=vocab_size)
token.fit_on_texts(sentence)

print(train_sentence[:5])
train_sequence = token.texts_to_sequences(train_sentence)
valid_sequence = token.texts_to_sequences(test_sentence)
print(train_sequence[:5])

trunc_type = 'post'
padding_type = 'post'
max_length = 120
train_pad = pad_sequences(train_sequence, truncating=trunc_type, padding=padding_type, maxlen=max_length)
valid_pad = pad_sequences(valid_sequence, truncating=trunc_type, padding=padding_type, maxlen=max_length)

train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

embedding_dim = 64
model = tf.keras.models.Sequential([
    Embedding(vocab_size, embedding_dim),
    Bidirectional(LSTM(64, return_sequences='True')),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_pad, train_labels, validation_data=(valid_pad, test_labels), epochs=3)

