import tensorflow as tf
import os
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv('C:\\Users\\82104\\PycharmProjects\\pythonProject\\TEST\\data\\heart.csv')
data = data.dropna()

ydata = data['HeartDisease'].values
xdata = data.iloc[:, :-1]

xdata['Sex'] = xdata['Sex'].map({'M': 0, 'F': 1})
xdata['ChestPainType'] = xdata['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
xdata['RestingECG'] = xdata['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2, 'TA': 3})
xdata['ExerciseAngina'] = xdata['ExerciseAngina'].map({'N': 0, 'Y': 1})
xdata['ST_Slope'] = xdata['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

xdata = np.array(xdata)
ydata = np.array(ydata)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2)
def heart_model():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(128, activation='relu'),
                                        tf.keras.layers.Dense(256, activation='relu'),
                                        tf.keras.layers.Dropout(0.3),
                                        tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(1, activation='sigmoid')
                                        ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = heart_model()

model.fit(x_train, y_train, epochs=100)
result = model.evaluate(x_test, y_test, verbose=2)

patient = [[40, 0, 0, 140, 289, 0, 0, 172, 0, 0, 0]]
pred = model.predict(patient)
pred = int(pred)

print("이환자가 암에걸릴 확률은 {:.2f}이다".format(pred)
)
print(result)
