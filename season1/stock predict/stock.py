import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
import datetime

data = pd.read_csv('dataset/005930.KS_5y.csv')

high_prices=data['High'].values
low_prices=data['Low'].values
mid_prices=(high_prices+low_prices)/2

seq_len = 50
sequence_length = seq_len+1

result = []
for idx in range(len(mid_prices)-sequence_length):
    result.append(mid_prices[idx: idx+sequence_length])

normalized_data = []

for window in result:
    normalized_window=[((float(p)/float(window[0]))-1)for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

row = int(round(result.shape[0]*0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=10, epochs=20)

pred = model.predict(x_test)

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='prediction')
ax.legend()
plt.show()