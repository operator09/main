import os
import pandas as pd
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv('gpascore.csv')
data = data.dropna()

ydata = data['admit'].values
xdata = []

for i, rows in data.iterrows():
    xdata.append([ rows['gre'], rows['gpa'], rows['rank']])

model = tf.keras.models.Sequential([tf.keras.layers.Dense(64,activation='sigmoid'),
                                    tf.keras.layers.Dense(128,activation='sigmoid'),
                                    tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(np.array(xdata), np.array(ydata), epochs=100000)

predict = model.predict([[750, 3.7, 3], [400, 2, 2.1]])
print(predict)