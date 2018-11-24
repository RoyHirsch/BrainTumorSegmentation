import scipy.io as spio
import numpy as np
import os
import re
import sys

ROOT_DIR = os.path.realpath(__file__ + "/../../")

def get_image_from_mat_file(dir =""):
    mat = spio.loadmat(dir, squeeze_me=True)
    im = mat['im']
    # optional values
    # volnames = mat['im_volnames']
    # filenames = mat['im_filenames']
    # resolution = mat['im_resolution']
    return im

def get_labels_from_mat_file (dir =""):
    mat = spio.loadmat(dir, squeeze_me=True)
    img = mat['gt4']
    return img

def get_data_and_labels_from_folder(dir = ROOT_DIR+'/Data'):
    """
        Extracts all the BARTS data and labels

        Creates a list with N entries (number of samples).
        In each cell is a 4D numpy.ndarray object with size: [H,W,D,C]
            H - image height
            W - image wights
            D - image depth
            C - channels / modalities
        dir - a root folder where all the data is in, the function scans the root folder.
    """
    Xtrain = []
    ytrain = []

    files = sorted(os.listdir(dir))[1:]
    for folder in files:
        fullPath = os.path.join(dir, folder)
        dataPath = os.path.join(fullPath, 'dataBN.mat')
        labelPath = os.path.join(fullPath, 'gt4.mat')
        img = get_image_from_mat_file(dataPath)
        Xtrain.append(img)
        labelImg = get_labels_from_mat_file(labelPath)
        ytrain.append(labelImg)

    # for root, dirs, files in os.walk(dir):
    #     for fileName in files:
    #         # logging.info(filename)
    #         match = re.search(r'dataBN', fileName)
    #         if match:
    #             img = get_image_from_mat_file(os.path.join(root, fileName))
    #             Xtrain.append(img)
    #         match = re.search(r'gt4', fileName)
    #         if match:
    #             labelImg = get_labels_from_mat_file(os.path.join(root, fileName))
    #             ytrain.append(labelImg)
    return Xtrain, ytrain

def get_single_mode_data(dataList, modality, isNorm):
    singleModeDataList = []
    for img in dataList:
        temp = img[:, :, :, modality]
        if isNorm:
            maxTemp = np.max(temp)
            temp = temp / maxTemp
        singleModeDataList.append(temp)
    return singleModeDataList

def get_shapes(list):
    for item in list:
        print(np.shape(item))


