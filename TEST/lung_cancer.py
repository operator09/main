import os
import numpy as np
import tensorflow as tf
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('C:\\Users\\operator09\\PycharmProjects\\Main\\TEST\\data\\survey lung cancer.csv')

df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 2})
x = df.drop('LUNG_CANCER', axis=1)
y = df.LUNG_CANCER
y = y.map({'YES': 1, 'NO': 2})

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                    ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(np.array(x_train), np.array(y_train), validation_data=(np.array(x_valid), np.array(y_valid)), epochs=5)
model.evaluate(x_valid, y_valid, verbose=2)

pred = [[1, 23, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2]]
predition = model.predict(pred)
print(predition)