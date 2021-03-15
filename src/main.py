from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import datetime

# TODO: move to config
# GLOBALS
from tensorflow.python.keras.callbacks import CSVLogger

TRAIN_DATASET_PATH = "data/train"
VAL_DATASET_PATH = "data/val"
TEST_DATASET_PATH = "data/test"
OUTPUT_PATH = "output/"
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def load_data(_directory: str):
    dataset = keras.preprocessing.image_dataset_from_directory(_directory, labels="inferred", batch_size=BATCH_SIZE,
                                                               image_size=IMAGE_SIZE)
    return dataset


def data_preprocessing(dataset):
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = dataset.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds


if __name__ == '__main__':
    dt = datetime.datetime.now()
    # AUGMENTATION
    #data_augmentation_rotation = tf.keras.Sequential([
    #    layers.experimental.preprocessing.RandomRotation(0.2),
    #])

    csv_logger = CSVLogger('training{dt}.log'.format(dt=dt))

    # LOAD DATA
    train_ds = load_data(TRAIN_DATASET_PATH)
    val_ds = load_data(VAL_DATASET_PATH)
    test_ds = load_data(TEST_DATASET_PATH)

    #augmentation_ds = train_ds.take(int(len(train_ds)*0.2))
    #augmentation_ds = augmentation_ds.map(lambda x, y: (data_augmentation_rotation(x, training=True), y))

    #train_ds = train_ds.concatenate(augmentation_ds)

    # train_ds = train_ds.cache().prefetch(buffer_size=8)
    # val_ds = val_ds.cache().prefetch(buffer_size=8)
    # test_ds = test_ds.cache().prefetch(buffer_size=8)

    # DATA PREPROCESSING
    normalized_train_ds = data_preprocessing(train_ds)
    normalized_val_ds = data_preprocessing(val_ds)
    normalized_test_ds = data_preprocessing(test_ds)

    # MODEL
    model = ResNet50(weights=None)
    #model = EfficientNetB0(weights=None)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
    history = model.fit(normalized_train_ds, epochs=2, validation_data=normalized_val_ds, verbose=1, batch_size=BATCH_SIZE, callbacks=[csv_logger])
    model.save()
    evaluate_result = model.evaluate(normalized_test_ds)
    print("\n\nTest loss:", evaluate_result[0], "Test accuracy: ", evaluate_result[1])
