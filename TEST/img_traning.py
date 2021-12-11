import os
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_train, ds_info = tfds.load('cats_vs_dogs', split=[tfds.Split.TRAIN], with_info=True)

img = [one['image'].numpy() for one in data_train[0].take(30)]

resnet50_pre = tf.keras.applications.resnet.ResNet50(weights='imagenet', input_shape = (224,224,3))

from tensorflow.keras.applications.imagenet_utils import decode_predictions

def pred_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img_resized=cv2.resize(img,(224,224))
    pred = resnet50_pre.predict(img_resized.reshape([1,224,224,3]))
    decode_pred = decode_predictions(pred)

    for i, instance in enumerate(decode_pred[0]):
        print("{}위:{} ({:2f}%)".format(i+1,instance[1],instance[2]*100))

print(pred_img(img[11]))