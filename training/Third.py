import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_img, train_label), (test_img, test_label) = fashion_mnist.load_data()

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_img = train_img/255
test_img = test_img/255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_img, train_label, epochs=10)

test_loss, test_acc = model.evaluate(test_img,  test_label, verbose=2)

print('\nTest accuracy:', test_acc)