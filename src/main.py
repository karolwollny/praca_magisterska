import datetime
import os
import png
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.callbacks import CSVLogger
from numpy import savetxt
import itertools
from consts import *
from augmentations import *
import argparse

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
    dataset = keras.preprocessing.image_dataset_from_directory(_directory, labels="inferred",
                                                               label_mode="int",
                                                               batch_size=BATCH_SIZE,
                                                               image_size=IMAGE_SIZE)
    return dataset


def data_preprocessing(dataset):
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = dataset.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(80, 80))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentation_type',
                        dest='augmentation_type',
                        type=str,
                        help='type of augmentation from list: none, rotation, zoom, noise, shift, crop, flip',
                        default='none',
                        choices=['none', 'rotation', 'zoom', 'noise', 'shift', 'crop', 'flip'],
                        )

    args = parser.parse_args()
    augmentation_type = args.augmentation_type

    dt = datetime.datetime.now()
    output_path = OUTPUT_PATH.format(a_type=augmentation_type, dt=dt)
    CHECKPOINT_PATH = os.path.join(output_path, CHECKPOINT_DIR_NAME, "cp.ckpt")
    os.mkdir(output_path)

    csv_logger = CSVLogger(os.path.join(output_path, 'training{dt}.log'.format(dt=dt)))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, verbose=1)

    # LOAD DATA
    train_ds = load_data(TRAIN_DATASET_PATH)
    val_ds = load_data(VAL_DATASET_PATH)
    test_ds = load_data(TEST_DATASET_PATH)

    #augmentation_ds = train_ds.take(int(len(train_ds)*0.2))
    #augmentation_ds = augmentation_ds.map(lambda x, y: (data_augmentation_rotation(x, training=True), y))

    #train_ds = train_ds.concatenate(augmentation_ds)

    # DATA PREPROCESSING
    normalized_train_ds = data_preprocessing(train_ds)
    normalized_val_ds = data_preprocessing(val_ds)
    normalized_test_ds = data_preprocessing(test_ds)

    test_labels = np.array([])

    for x, y in normalized_test_ds:
        test_labels = np.concatenate((test_labels, y.numpy()))

    # MODEL
    model = ResNet50(weights=None, classes=100)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy', 'Hinge'])

    history = model.fit(normalized_train_ds,
                        epochs=EPOCHS,
                        validation_data=normalized_val_ds,
                        verbose=1,
                        batch_size=BATCH_SIZE,
                        callbacks=[csv_logger, checkpoint_callback])

    model.save(os.path.join(output_path, 'model_{dt}.h5'.format(dt=dt)))

    evaluate_result = model.evaluate(normalized_test_ds)
    print("\n\nTest loss:", evaluate_result[0], "Test accuracy: ", evaluate_result[1])

    pred = model.predict(normalized_test_ds)
    pred_max = []
    for x in pred:
        pred_max.append(np.argmax(x))
    np.set_printoptions(precision=0)
    cm = tf.math.confusion_matrix(test_labels, pred_max).numpy()
    cm_img = plot_confusion_matrix(cm, set(test_labels.astype(np.int32)))
    plt.savefig(os.path.join(output_path, 'cm_img_{dt}.png'.format(dt=dt)))
    np.savetxt(os.path.join(output_path, 'cm_{dt}.csv'.format(dt=dt)), cm)

    plt.figure()

    plt.plot(EPOCHS, history.history['accuracy'], 'bo', label='Training acc')
    plt.plot(EPOCHS, history.history['val_accuracy'], 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'accuracy_history_img_{dt}.png'.format(dt=dt)))

    plt.figure()

    plt.plot(EPOCHS, history.history['loss'], 'bo', label='Training loss')
    plt.plot(EPOCHS, history.history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.savefig(os.path.join(output_path, 'loss_history_img_{dt}.png'.format(dt=dt)))

