import datetime

import tensorflow as tf
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
import scipy.ndimage as ndimage

from tensorflow.keras.applications.resnet50 import ResNet50

import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["00_dog",
           "01_bird",
           "02_wheeled vehicle",
           "03_reptile",
           "04_carnivore",
           "05_insect",
           "06_musical instrument",
           "07_primate",
           "08_fish"]

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

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
)

_img_6 = keras.preprocessing.image.load_img('/home/karol/Desktop/uczelnia/Magisterka/data/test/07_primate/n02492035_21520.JPEG',target_size=(224,224))
_img_5 = keras.preprocessing.image.load_img('/home/karol/Desktop/uczelnia/Magisterka/data/test/07_primate/n02483708_20341.JPEG',target_size=(224,224))
_img_4 = keras.preprocessing.image.load_img('/home/karol/Desktop/uczelnia/Magisterka/data/test/00_dog/n02096051_44543.JPEG',target_size=(224,224))
_img_3 = keras.preprocessing.image.load_img('/home/karol/Desktop/uczelnia/Magisterka/data/test/00_dog/n02102973_39515.JPEG',target_size=(224,224))
_img_2 = keras.preprocessing.image.load_img('/home/karol/Desktop/uczelnia/Magisterka/data/test/00_dog/n02096051_09703.JPEG',target_size=(224,224))
_img_1 = keras.preprocessing.image.load_img('/home/karol/Desktop/uczelnia/Magisterka/data/test/00_dog/n02100583_45217.JPEG',target_size=(224,224))

_imgs = [(_img_1, '00_dog'), (_img_2, '00_dog'), (_img_3, '00_dog'), (_img_4, '00_dog'), (_img_5, '07_primate'),
         (_img_6, '07_primate')]

#_img = keras.preprocessing.image.load_img('/home/karol/Desktop/uczelnia/Magisterka/data/test/06_musical instrument/n04536866_45057.JPEG',target_size=(224,224))

# plt.imshow(_img)
# plt.show()

model = tf.keras.models.load_model('/home/karol/Desktop/wyniki/image-net-9/bez_augmentacji/model_augmentation_none_2021-05-04 00:49:39.141861/model_2021-05-04 00:49:39.141861.h5', compile=True)
#model = tf.keras.models.load_model('/home/karol/Desktop/wyniki/image-net-9/augmentacja_na_doklejonym_zbiorze_x2_multi /model_augmentation_multi_2021-05-05 05:58:47.998631/model_2021-05-05 05:58:47.998631.h5', compile=True)

# img = keras.preprocessing.image.img_to_array(_img)
# img = tf.expand_dims(img, 0)
# img = img / 255
# y_pred = model.predict(img)
# images = tf.Variable(img, dtype=float)
#
# with tf.GradientTape() as tape:
#     pred = model(images, training=False)
#     class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
#     loss = pred[0][class_idxs_sorted[0]]
#
# grads = tape.gradient(loss, images)
# dgrad_abs = tf.math.abs(grads)
# dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
#
# arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
#
# grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
fig, axes = plt.subplots(3, 4, figsize=(10, 10))


def get_pred(_img):
    img = keras.preprocessing.image.img_to_array(_img)
    img = tf.expand_dims(img, 0)
    img = img / 255
    y_pred = model.predict(img)
    images = tf.Variable(img, dtype=float)

    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]

    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)

    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    return y_pred, grad_eval


num = 0
for a in axes:
    y_pred, grad_eval = get_pred(_imgs[num][0])

    a[0].imshow(_imgs[num][0])
    a[0].set_title(f"Prawdziwa klasa:\n{_imgs[num+1][1]}")
    i = a[1].imshow(grad_eval,cmap="jet",alpha=0.8)

    gaus = ndimage.gaussian_filter(grad_eval, sigma=5)
    a[1].imshow(_imgs[num][0])
    first_pred = f"{CLASSES[tf.argmax(y_pred, axis=1)[0]]} - {'%.2f' % float(np.max(y_pred)*100)}%"
    second_pred = f"{CLASSES[tf.nn.top_k(y_pred, k=2).indices[0][1]]} - " \
                  f"{'%.2f' % float(tf.nn.top_k(y_pred, k=2).values[0][1]*100)}%"

    a[1].set_title(f"Predykcja:\n{first_pred}\n{second_pred}")
    a[1].imshow(gaus, alpha=.65)

    a[2].imshow(_imgs[num+1][0])
    a[2].set_title(f"Prawdziwa klasa:\n{_imgs[num+1][1]}")
    i = a[3].imshow(grad_eval, cmap="jet", alpha=0.8)

    gaus = ndimage.gaussian_filter(grad_eval, sigma=5)
    a[3].imshow(_imgs[num+1][0])
    first_pred = f"{CLASSES[tf.argmax(y_pred, axis=1)[0]]} - {'%.2f' % float(np.max(y_pred)*100)}%"
    second_pred = f"{CLASSES[tf.nn.top_k(y_pred, k=2).indices[0][1]]} - " \
                  f"{'%.2f' % float(tf.nn.top_k(y_pred, k=2).values[0][1]*100)}%"

    a[3].set_title(f"Predykcja:\n{first_pred}\n{second_pred}")
    a[3].imshow(gaus, alpha=.65)

    num += 2

#plt.colorbar(i)
plt.tight_layout()
#img = plt.show()

#plt.show()
dt = datetime.datetime.now()
plt.savefig('sm{dt}.png'.format(dt=dt))
