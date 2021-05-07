from tensorflow.keras import layers
import tensorflow as tf


def image_augmentation_rotation():
    print("AUGMENTATION TYPE: ROTATION")
    return tf.keras.Sequential([
        layers.experimental.preprocessing.RandomRotation((-0.25, 0.25)),
    ])


def image_augmentation_zoom():
    print("AUGMENTATION TYPE: ZOOM")
    return tf.keras.Sequential([
        layers.experimental.preprocessing.RandomZoom(
        height_factor=(-0.2, 0.5), width_factor=(-0.2, 0.5), fill_mode='reflect',
        interpolation='bilinear'),
    ])


def image_augmentation_noise():
    print("AUGMENTATION TYPE: NOISE")
    return tf.keras.Sequential([
        layers.GaussianNoise(0.75),
    ])


def image_augmentation_shift():
    print("AUGMENTATION TYPE: SHIFT")
    pass


def image_augmentation_crop():
    # PRZED PRZEPROWADZENIEM TESTÓW NA TEJ AUGMENTACJI, TRZEBA ZMIENIC ROZMIAR PLIKOW PRZY ŁADOWANIU DANYCH DO DATASETU!
    print("AUGMENTATION TYPE: CROP")
    return tf.keras.Sequential([
        layers.experimental.preprocessing.RandomCrop(224, 224)
    ])


def image_augmentation_flip():
    print("AUGMENTATION TYPE: FLIP")
    return tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    ])


def image_multi_augmentation():
    print("AUGMENTATION TYPE: MULTI")
    return tf.keras.Sequential([
        layers.experimental.preprocessing.RandomRotation((-0.25, 0.25)),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    ])
