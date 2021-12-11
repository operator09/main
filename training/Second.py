import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0 , x_test / 255.0

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), #입력이미지 평평하게
                             tf.keras.layers.Dense(128, activation='sigmoid'),
                             tf.keras.layers.Dropout(0.2), #과적합 방지용 데이터를 좀 떨군다 0.2 = 20%지운다
                             tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4)
result = model.evaluate(x_test,  y_test, verbose=2) #모델평가 [0]정답률 [1] 손실률
print(result)