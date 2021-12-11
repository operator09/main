import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

df = pd.read_csv("C:\\Users\\82104\\PycharmProjects\\pythonProject\\TEST\\data\\divorce_data.csv",delimiter=';')

y = df['Divorce']
x = df.drop('Divorce', axis=1)
x, y = np.array(x), np.array(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu',),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(248, activation='relu'),
                                    tf.keras.layers.Dropout(0.3),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
])
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("loss: {}, acc: {}".format(loss*100, acc*100))

tlist = x_test[[0]]
print(tlist)
pred = model.predict(tlist)
pred = float(pred)
print(pred)
print("당신이 헤어질 확률은 {:.2f}이다".format(pred))

