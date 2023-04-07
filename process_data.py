import numpy as np
from scipy import ndimage
import os


def preprocessing_volume(original_image, new_x, new_y, new_z):
    """
    Função para pré processar as imagens, padronizando-as em valores entre
    0 e 1, e garantir uma quantidade fixa para os eixos xyz.
    """
    dims = original_image.shape
    new_image = ndimage.zoom(original_image, zoom=[new_x/dims[0], new_y/dims[1], new_z/dims[2], 1])

    if need_rotation(original_image):
      new_image = ndimage.rotate(new_image, 180, axes=(0, 1))

    return new_image[:,:,:,:]


def create_features_dataset_from_npy(file_path, save_path, new_x, new_y, new_z):
  """
  Função para ler os arquivos .npy das imagens originais e preprocessá-las para gerar novos .npy
  """

  dirs = os.listdir(file_path)
  for dir in dirs[1:]:
      if not os.path.exists(f"{save_path}/{dir}"):
          try:
              i = 0
              image = np.load(f"{file_path}/{dir}")
              print(image.shape)
              if new_z == -1:
                image = preprocessing_volume(image, new_x, new_y, image.shape[-2])
              else:
                image = preprocessing_volume(image, new_x, new_y, new_z)

              np.save(f"{save_path}/{dir}", image)
              print(save_path + "/" + dir + " was created")
          except:
            print("error on: " + file_path + "/" + dir)


if __name__ == '__main__':
    X = 175
    Y = 260
    Z = 10

    create_features_dataset_from_npy('WHERE DATA ARE', 'WHERE NEW DATA GOES', X, Y, Z)
