import numpy as np
import pandas as pd
import os
import time
import random
from matplotlib.image import imsave
import matplotlib.pyplot as plt
from scipy import ndimage


def separate_slices(img):
    slices = []

    for i in range(img.shape[-2]):
        slices.append(img[:, :, i])

    slices.append(np.mean(img, axis=-2))

    return np.array(slices)


def get_labels(label_file):
    """lê a tabela com as informações dos pacientes e retorna uma matriz com o ID e as labels"""
    labels = pd.read_csv(label_file)
    cancer_labels = dict()

    for p in labels.index:
        cancer_labels[labels['PatientID'][p]] = [int(labels['Normal'][p]), int(labels['Actionable'][p]),
                                                 int(labels['Benign'][p]), int(labels['Cancer'][p])]

    return cancer_labels


def flip_image(x):
    return x[:, x.shape[1]-1:0:-1]


def separate_slices(img):
    slices = []

    for i in range(img.shape[-2]):
        slices.append(img[:, :, i])

    slices.append(np.mean(img, axis=-2).astype(dtype='float16'))

    return np.array(slices)


def data_augmentation(x):
    new_images = []

    new_images.append(x.astype('float16'))

    new_images.append(np.squeeze(
        ndimage.rotate(np.expand_dims(x, -1).astype('float32'), angle=10, axes=(1, 0), reshape=False).astype(
            'float16'), -1))
    new_images.append(np.squeeze(
        ndimage.rotate(np.expand_dims(x, -1).astype('float32'), angle=-10, axes=(1, 0), reshape=False).astype(
            'float16'), -1))

    x = flip_image(x).astype('float16')
    new_images.append(x)
    new_images.append(np.squeeze(
        ndimage.rotate(np.expand_dims(x, -1).astype('float32'), angle=10, axes=(1, 0), reshape=False).astype(
            'float16'), -1))
    new_images.append(np.squeeze(
        ndimage.rotate(np.expand_dims(x, -1).astype('float32'), angle=-10, axes=(1, 0), reshape=False).astype(
            'float16'), -1))

    return new_images


if __name__ == '__main__':
    base_dir = 'C:\\Users\\Gabriel\\Downloads\\'
    images = os.listdir(base_dir + 'images')
    labels = get_labels(base_dir + 'labels.csv')

    counter = 0
    for i in range(len(images)):
        start = time.time()
        image3d = np.load(base_dir + 'images\\' + images[i])
        slices = separate_slices(image3d)
        rand = random.random()
        for j, img in enumerate(slices):
            img = np.squeeze(img, -1)
            if labels[images[i][:-4]] == [1, 0, 0, 0]:
                if rand > 0.33:
                    img = data_augmentation(img)
                    for k, new_image in enumerate(img):
                        imsave(f'C:\\Users\\Gabriel\\Desktop\\Normal\\image_{i}_{j}_{k}.png', new_image)
                elif rand > 0.2:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Validation_Normal\\image_{i}_{j}.png', img)
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Validation_Normal\\image_{i}_{j}_f.png', flip_image(img))
                else:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Test_Normal\\image_{i}_{j}.png', img)
            elif labels[images[i][:-4]] == [0, 1, 0, 0]:
                if rand > 0.33:
                    img = data_augmentation(img)
                    for k, new_image in enumerate(img):
                        imsave(f'C:\\Users\\Gabriel\\Desktop\\Actionable\\image_{i}_{j}_{k}.png', new_image)
                elif rand > 0.2:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Validation_Actionable\\image_{i}_{j}.png', img)
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Validation_Actionable\\image_{i}_{j}_f.png', flip_image(img))
                else:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Test_Actionable\\image_{i}_{j}.png', img)
            elif labels[images[i][:-4]] == [0, 0, 1, 0]:
                if rand > 0.33:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Benign\\image_{i}_{j}.png', img)
                elif rand > 0.2:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Validation_Benign\\image_{i}_{j}.png', img)
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Validation_Benign\\image_{i}_{j}_f.png', flip_image(img))
                else:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Test_Benign\\image_{i}_{j}.png', img)
            else:
                if rand > 0.33:
                    img = data_augmentation(img)
                    for k, new_image in enumerate(img):
                        imsave(f'C:\\Users\\Gabriel\\Desktop\\Cancer\\image_{i}_{j}_{k}.png', new_image)
                elif rand > 0.2:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Validation_Cancer\\image_{i}_{j}.png', img)
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Validation_Cancer\\image_{i}_{j}_f.png', flip_image(img))
                else:
                    imsave(f'C:\\Users\\Gabriel\\Desktop\\Test_Cancer\\image_{i}_{j}.png', img)

        print(time.time() - start)
        counter += 1
    print(counter)
