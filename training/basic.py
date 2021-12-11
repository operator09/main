import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

height = 170
size = 280
a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss():
    pre = height * a + b
    return tf.square(size-pre)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

while True:
    opt.minimize(loss, var_list=[a,b])
    pred = height * a + b
    print(a.numpy(), b.numpy())
    if size == pred:
        print(pred.numpy())
        break