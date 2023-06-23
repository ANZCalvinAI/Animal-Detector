import os
import logging
import cv2
from utils import (
    fnGet_dataset_dict,
    fnGenerate_augimg
)


const_classesFile = "models/labels.txt"
const_modelWeights = "models/yolov5x.onnx"
const_dataset_path = "./dataset"


def load_deepLearning():

    # Load class names.
    classes = None
    with open(const_classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the weight files to the model and load the network using them.
    net = cv2.dnn.readNet(const_modelWeights)

    return net, classes

def extract_label(species_folder_name):

    return species_folder_name


if __name__ == '__main__':
    '''
    Enhance training dataset.
    Prepare ./models before running.
    models
    ├── labels.txt
    └── yolov5l.onnx

    '''

    net, classes = load_deepLearning()

    # Get image list.
    data_dict = fnGet_dataset_dict(const_dataset_path)
    for k_label, v_images in data_dict.items():

        print(f"\nk_label: {k_label}")
        for image_path in v_images:

            print(f"\n\tLoad image: {image_path}")
            img = cv2.imread(image_path)
            # gt = extract_label(k_label)
            gt = "Insecta"
            aug_output = fnGenerate_augimg(img, net, classes, gt)
            print("\t[augment] Generate {} more.".format(len(aug_output)))





