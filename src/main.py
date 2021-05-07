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
import sklearn.metrics


# TODO: augmentacje

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
    dataset = keras.preprocessing.image_dataset_from_directory(_directory,
                                                               labels="inferred",
                                                               label_mode="categorical",
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
    figure = plt.figure(figsize=(9, 9))
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
                        choices=['none', 'rotation', 'zoom', 'noise', 'shift', 'crop', 'flip', 'multi'],
                        )

    args = parser.parse_args()
    augmentation_type = args.augmentation_type

    dt = datetime.datetime.now()
    output_path = OUTPUT_PATH.format(a_type=augmentation_type, dt=dt)
    CHECKPOINT_PATH = os.path.join(output_path, CHECKPOINT_DIR_NAME, "cp.ckpt")
    os.mkdir(output_path)

    csv_logger = CSVLogger(os.path.join(output_path, 'training{dt}.log'.format(dt=dt)))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, verbose=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # LOAD DATA
    train_ds = load_data(TRAIN_DATASET_PATH)
    val_ds = load_data(VAL_DATASET_PATH)
    test_ds = load_data(TEST_DATASET_PATH)

    if augmentation_type != 'none':
        if augmentation_type == 'multi':
            augmentation = image_multi_augmentation()
        elif augmentation_type == 'rotation':
            augmentation = image_augmentation_rotation()
        elif augmentation_type == 'zoom':
            augmentation = image_augmentation_zoom()
        elif augmentation_type == 'noise':
            augmentation = image_augmentation_noise()
        elif augmentation_type == 'shift':
            augmentation = image_augmentation_shift()
        elif augmentation_type == 'crop':
            augmentation = image_augmentation_crop()
        elif augmentation_type == 'flip':
            augmentation = image_augmentation_flip()

        augmentation_ds = train_ds
        augmentation_ds = augmentation_ds.map(lambda x, y: (augmentation(x, training=True), y))

        train_ds = train_ds.concatenate(augmentation_ds)

    # DATA PREPROCESSING
    normalized_train_ds = data_preprocessing(train_ds)
    normalized_val_ds = data_preprocessing(val_ds)
    normalized_test_ds = data_preprocessing(test_ds)

    # MODEL
    # model = ResNet50(weights=None, classes=9)
    model = tf.keras.models.load_model('./data/model_9_classes_resnet.h5', compile=False)
    #model = tf.keras.models.load_model('/home/karol/Desktop/wyniki/image-net-9/bez_augmentacji/model_augmentation_none_2021-05-04 00:49:39.141861/model_2021-05-04 00:49:39.141861.h5', compile=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy', 'Precision', 'AUC', 'Recall'])

    history = model.fit(normalized_train_ds,
                        epochs=EPOCHS,
                        validation_data=normalized_val_ds,
                        verbose=1,
                        batch_size=BATCH_SIZE,
                        callbacks=[csv_logger, checkpoint_callback, early_stopping_callback])

    model.save(os.path.join(output_path, 'model_{dt}.h5'.format(dt=dt)))

    evaluate_result = model.evaluate(normalized_test_ds)
    print("\n\nTest loss:", evaluate_result[0], "Test accuracy: ", evaluate_result[1])

    _image_predict = []
    _labels = []

    for _image, _label in normalized_test_ds:
        _image_predict = tf.concat([_image_predict, tf.argmax(model.predict(_image), axis=1)], axis=0)
        _labels = tf.concat([_labels, tf.argmax(_label, axis=1)], axis=0)

    print(sklearn.metrics.classification_report(_labels, _image_predict))
    print(sklearn.metrics.f1_score(_labels, _image_predict, average=None))

    cm = tf.math.confusion_matrix(_labels, _image_predict).numpy()
    cm_img = plot_confusion_matrix(cm, test_ds.class_names)
    # cm_img = plot_confusion_matrix(cm, set(_labels.numpy().astype(np.int32)))
    plt.savefig(os.path.join(output_path, 'cm_img_{dt}.png'.format(dt=dt)), bbox_inches = "tight")
    np.savetxt(os.path.join(output_path, 'cm_{dt}.csv'.format(dt=dt)), cm)

    plt.figure()

    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_path, 'accuracy_history_img_{dt}.png'.format(dt=dt)))

    plt.figure()

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_path, 'loss_history_img_{dt}.png'.format(dt=dt)))

