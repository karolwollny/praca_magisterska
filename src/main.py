from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np

# TODO: move to config
# GLOBALS
TRAIN_DATASET_PATH = "data/train"
VAL_DATASET_PATH = "data/val"


def load_data(_directory: str):
    dataset = keras.preprocessing.image_dataset_from_directory(_directory, batch_size=64, image_size=(200, 200))

    # for data, labels in dataset:
    #     print(data.shape)
    #     print(data.dtype)
    #     print(labels.shape)
    #     print(labels.dtype)

    return dataset


def data_preprocessing(dataset):
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = dataset.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds


if __name__ == '__main__':
    # LOAD DATA
    train_ds = load_data(TRAIN_DATASET_PATH)
    val_ds = load_data(VAL_DATASET_PATH)
    # DATA PREPROCESSING
    normalized_train_ds = data_preprocessing(train_ds)

    train_ds = train_ds.prefetch(buffer_size=32)

    # MODEL
    model = ResNet50(weights=None)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, epochs=5, validation_data=val_ds,)