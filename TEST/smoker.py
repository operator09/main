import pandas as pd
import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

df = pd.read_csv('C:\\Users\\82104\\PycharmProjects\\pythonProject\\TEST\\data\\smoking.csv')

df['outcome'] = df['outcome'].map({'Alive': 0, 'Dead': 1})
df['smoker'] = df['smoker'].map({'Yes': 0, 'No': 1})

xdata = df.drop('outcome', axis=1)
ydata = df['outcome']
xdata = np.array(xdata)
ydata = np.array(ydata)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid'),
])
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000)
loss , acc = model.evaluate(x_test, y_test)
print("loss : {}, acc : {}".format(loss, acc))