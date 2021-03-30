from tensorflow.keras import layers
import tensorflow as tf


def image_augmentation_rotation():
    return tf.keras.Sequential([
       layers.experimental.preprocessing.RandomRotation(0.2),
    ])


def image_augmentation_zoom():
    pass


def image_augmentation_noise():
    pass


def image_augmentation_shift():
    pass


def image_augmentation_crop():
    pass


def image_augmentation_flip():
    pass
