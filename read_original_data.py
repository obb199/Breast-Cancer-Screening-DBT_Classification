import pydicom
import numpy as np
import pandas as pd
from scipy import ndimage
import os


def read_dicom_image(fp):
    """
    Função para ler arquivo dcm, retornando a imagem e o ID do paciente

    fp: caminho da pasta que contem o arquivo original da iamgem (.dicom)
    """
    frames = sorted([f for f in os.listdir(fp) if f[-4:] == '.dcm'])
    if len(frames) == 0:
        return
    image = []
    for frame in frames:
        data = pydicom.dcmread(fp + '/' + frame)
        image.append(data.pixel_array)

    return np.transpose(image) / 1023.0, fp[64:74]


def get_dicom_dirs(d='BCS-DBT/Breast-Cancer-Screening-DBT/'): #REMEMBER TO CHANGE HERE!!!
    """
    Função para criar uma lista com todos os filepaths das imagens

    d: pasta que possui os arquivos originais.
    """

    dirs = []

    for f in os.listdir(d)[1:]:
        f1 = os.listdir(d + f)
        f2 = os.listdir(d + str(f) + '/' + str(f1[0]))
        dirs.append(d + str(f) + '/' + str(f1[0]) + '/' + str(f2[0]))

    return dirs


def need_rotate(img):
    """
    Função para verificar se há necessidade de rotacionar 180 graus a imagem.
    """
    if np.sum(img[0:img.shape[0]//2, :, :, :]) > np.sum(img[img.shape[0]//2:, :, :, :]):
        return True
    return False


def create_features_dataset_from_dicom(save_path):
    """
    rotina para ler os arquivos originais e gerar novos arquivos .npy
    """
    dirs = get_dicom_dirs()
    for d in dirs:
        if os.path.exists(f'{save_path}/{d[64:74]}.npy'):
            print(f'file "{save_path}/{d[64:74]}.npy" exists')
            continue

        dicom, ID = read_dicom_image(d)
        np.save(f'{save_path}/{ID}', dicom)


if __name__ == '__main__':
    create_features_dataset_from_dicom('PATH OF DATASET HERE')
    #dataset -> https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580
