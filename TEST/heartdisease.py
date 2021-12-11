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

print(ydata.shape)
exit()
def heart_model():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(128, activation='relu'),
                                        tf.keras.layers.Dense(256, activation='relu'),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(1, activation='sigmoid')
                                        ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = heart_model()

model.fit(xdata, ydata, epochs=1000)
result = model.evaluate(xdata, ydata, verbose=2)

patient = [[40, 0, 0, 140, 289, 0, 0, 172, 0, 0, 0]]
pred = model.predict(patient)

print(pred)
print(result)
