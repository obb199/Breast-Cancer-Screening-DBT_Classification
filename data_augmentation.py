import numpy as np
from scipy import ndimage
from random import random, seed
from sklearn.utils import shuffle


def data_augmentation(X, y):
    new_images = []
    new_labels = []
    for x, label in zip(X, y):
        x = np.squeeze(x, -1).astype('float32')
        new_images.append(np.expand_dims(x, -1).astype('float16'))

        new_images.append(
            ndimage.rotate(np.expand_dims(x, -1).astype('float32'), angle=10, axes=(1, 0), reshape=False).astype(
                'float16'))
        new_images.append(
            ndimage.rotate(np.expand_dims(x, -1).astype('float32'), angle=-10, axes=(1, 0), reshape=False).astype(
                'float16'))

        x = cv2.flip(x, 1)
        new_images.append(np.expand_dims(x, -1).astype('float16'))
        new_images.append(
            ndimage.rotate(np.expand_dims(x, -1).astype('float32'), angle=10, axes=(1, 0), reshape=False).astype(
                'float16'))
        new_images.append(
            ndimage.rotate(np.expand_dims(x, -1).astype('float32'), angle=-10, axes=(1, 0), reshape=False).astype(
                'float16'))

        for _ in range(6):
            new_labels.append(label)

    return np.array(new_images), np.array(new_labels)