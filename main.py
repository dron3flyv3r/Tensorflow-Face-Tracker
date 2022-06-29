
print("importing libraries...")
import tensorflow as tf
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
print("import complete")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)
labels = tf.data.Dataset.list_files('data/labels/*.json', shuffle=False)

def load_image(x):
    byte_image = tf.io.read_file(x)
    img = tf.image.decode_jpeg(byte_image)
    return img

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()