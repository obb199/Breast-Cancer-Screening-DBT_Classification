import pandas as pd
import numpy as np
import os


def get_labels(label_file):
    """lê a tabela com as informações dos pacientes e retorna uma matriz com o ID e as labels"""
    labels = pd.read_csv(label_file)
    cancer_labels = dict()

    for p in labels.index:
        cancer_labels[labels['PatientID'][p]] = [int(labels['Normal'][p]), int(labels['Actionable'][p]), int(labels['Benign'][p]), int(labels['Cancer'][p])]

    return cancer_labels


def read_dataset(features_pathfile, csv_labels_pathfile, images_per_class):
    """
      A partir de um conjunto de numpy arrays salvos e do csv com as
    labels, será montado o conjunto de dados para uso nos algoritmos com um
    número máximo de features por classe.
    """

    examples_per_class = [0, 0, 0, 0]

    pacients_ID = os.listdir(features_pathfile)
    labels_list = get_labels(csv_labels_pathfile)
    X, y = [], []
    for ID in pacients_ID[1:]:
        label = labels_list[ID[:-4]]
        if images_per_class != -1:
            if examples_per_class[np.argmax(label)] > images_per_class:
                continue

            examples_per_class[np.argmax(label)] += 1
            X.append(np.load(features_pathfile + '/' + ID))

            if len(y) == images_per_class*4:
                break
        else:
            X.append(np.load(features_pathfile + '/' + ID))

        y.append(label)

    return np.array(X), np.array(y)


def multiply_labels(label, n_slices):
    new_labels = []

    for _ in range(n_slices + 1):
        new_labels.append(np.array(label, dtype='uint8'))

    return new_labels


def separate_slices(img):
    slices = []

    for i in range(img.shape[-2]):
        slices.append(img[:, :, i])

    slices.append(np.mean(img, axis=-2).astype(dtype='float16'))

    return slices


def read_dirs(path='/kaggle/input/320x240xdepth/'):
    dirs = []
    for d in os.listdir(path):
        dirs.append(path + '/' + d)

    return dirs


def separate_dirs(dirs, percent_train=0.6, percent_val=0.2, percent_test=0.2):
    length_dirs = len(dirs)

    train_dirs = dirs[0:int(percent_train * length_dirs)]
    val_dirs = dirs[int(percent_train * length_dirs):int((percent_val + percent_train) * length_dirs)]
    test_dirs = dirs[int((percent_val + percent_train) * length_dirs):]

    return train_dirs, val_dirs, test_dirs


def separate_data_into_groups(data, elements_per_group=20, shuffle_dirs=True, seed=-1):
    if shuffle_dirs:
        if seed == -1:
            data = shuffle(data, random_state=random.randint(0, 999999999))
        else:
            data = shuffle(data, random_state=seed)

    groups = []

    for i in range(0, len(data), elements_per_group):
        groups.append(data[i:i + elements_per_group])

    return groups

def create_2D_dataset(X, y):
  X, y = shuffle(X, y, random_state=437843)

  length = len(X)
  slices = X.shape[-2]

  X_train3D, y_train3D = X[0:int(length*0.6)], y[0:int(length*0.6)]
  X_val3D, y_val3D = X[int(length*0.6):int(length*0.8)], y[int(length*0.6):int(length*0.8)]
  X_test3D, y_test3D = X[int(length*0.8):], y[int(length*0.8):]

  X_train2D, X_val2D, X_test2D = [], [], []
  y_train2D, y_val2D, y_test2D = [], [], []


  for x, y in zip(X_train3D, y_train3D):
    X_train2D += separate_slices(x)
    y_train2D += multiply_labels(y, slices)

  for x, y in zip(X_val3D, y_val3D):
    X_val2D += separate_slices(x)
    y_val2D += multiply_labels(y, slices)

  for x, y in zip(X_test3D, y_test3D):
    X_test2D += separate_slices(x)
    y_test2D += multiply_labels(y_test3D, slices)

  return np.array(X_train2D), np.array(X_val2D), \
         np.array(X_test2D), np.array(y_train2D), \
         np.array(y_val2D), np.array(y_test2D)
  



if __name__ == '__main__':
    X, y = read_dataset('DIR OF IMAGES',
                        'DIR OF CSV WITH LABELS', '-1 or other number until 39')
