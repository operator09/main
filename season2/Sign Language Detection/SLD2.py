from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_PATH = os.path.join('MP_DATA')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequence = 30
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequence):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=1234)

log_dir = os.path.join("Logs")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

model = tf.keras.models.Sequential(
    [tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
     tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'),
     tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(32, activation='relu'),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(actions.shape[0], activation='softmax')

     ])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=2000, validation_data=(x_test, y_test), callbacks=[
        ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')])