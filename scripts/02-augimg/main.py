import os
import logging
import cv2
from utils import (
    fnGet_dataset_dict,
    fnGenerate_augimg
)


const_classesFile = "models/labels.txt"
const_modelWeights = "models/yolov5x.onnx"
const_dataset_path = "../../datasets/demo3-527classes/images"


def load_deepLearning():

    # Load class names.
    classes = None
    with open(const_classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the weight files to the model and load the network using them.
    net = cv2.dnn.readNet(const_modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

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
    label_idx = 0
    total_generated_idx = 0
    for k_label, v_images in data_dict.items():
        label_idx += 1
        print(f"\n[{label_idx}] k_label: {k_label}")
        for image_path in v_images:

            print(f"\n\tLoad image: {image_path}")
            img = cv2.imread(image_path)
            # gt = extract_label(k_label)
            gt = "Insecta"
            aug_output = fnGenerate_augimg(img, net, classes, gt)
            print("\t[augment] Generate {} more.".format(len(aug_output)))

            image_name = os.path.basename(image_path)
            image_dirPath = os.path.dirname(image_path)

            for idx, each_img in enumerate(aug_output):

                total_generated_idx += 1

                aug_image_path = os.path.join(image_dirPath, "aug{}_{}".format(idx, image_name))
                print("\t[new] Save {}th in {}".format(total_generated_idx, aug_image_path))
                cv2.imwrite(aug_image_path, each_img)

