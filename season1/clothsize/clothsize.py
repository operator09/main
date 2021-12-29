import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
"""
df = pd.read_csv('C:\\Users\\82104\\PycharmProjects\\pythonProject\\TEST\\data\\clothsize.csv')

df['age'] = df['age'].fillna(df['age'].median())
df['height'] = df['height'].fillna(df['height'].median())

"""
df.isna().sum()
""""""
df.columns
""""""
plt.style.use("seaborn")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x=df['size'],palette='hls')
sns.distplot(x=df['height'],color='r')
plt.show()
"""
"""
df["size"].value_counts()
"""
df['size'] = df['size'].map({'M': 1, 'S': 2, 'XXXL': 3, 'XL': 4, 'L': 5, 'XXS': 6, 'XXL': 7})

x = df.drop("size", axis=1)
"""
axis = 0은 행단위
axis = 1은 열단위
"""
y = df['size']
y = tf.keras.utils.to_categorical(y)

x = np.array(x)
y = np.array(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation='relu'),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(8, activation='softmax')
                                    ])


opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)
model.evaluate(x_test, y_test, verbose=2)

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
"""