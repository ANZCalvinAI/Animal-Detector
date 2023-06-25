import os
import random
import cv2
import logging
import numpy as np
from yolov5 import (
    FONT_FACE, FONT_SCALE, THICKNESS,
    # BLACK, YELLOW,
    RED, BLUE,
    pre_process,
    post_process,
    draw_label,
)


logging.getLogger().setLevel(logging.INFO)


def fnProjectiveTransform(img, distort_maxRatio):

    try:
        img_rows, img_cols = img.shape[:2]

        # ---------------------------------------------------------
        # 1.1 top left, top right, button right, button left.
        # ---------------------------------------------------------

        pts1 = np.float32([[0, 0], [img_cols-1, 0],
                           [img_cols-1, img_rows-1], [0, img_rows-1]])

        # ---------------------------------------------------------
        # 1.2 random policy.
        # ---------------------------------------------------------

        x_pad = int(img_cols * distort_maxRatio)
        y_pad = int(img_rows * distort_maxRatio)

        top_left_x = random.randint(0, x_pad)
        top_left_y = random.randint(0, y_pad)

        top_right_x = random.randint(img_cols-x_pad, img_cols)
        top_right_y = random.randint(0, y_pad)

        button_right_x = random.randint(img_cols-x_pad, img_cols)
        button_right_y = random.randint(img_rows-y_pad, img_rows)

        button_left_x = random.randint(0, x_pad-1)
        button_left_y = random.randint(img_rows-y_pad, img_rows)

        pts2 = np.float32([[top_left_x, top_left_y],
                           [top_right_x, top_right_y],
                           [button_right_x, button_right_y],
                           [button_left_x, button_left_y]])

        # ------------------------------------------------
        # 2.1 distort image and mask
        # size(cols, rows)
        # ------------------------------------------------

        M = cv2.getPerspectiveTransform(pts2, pts1)
        dst = cv2.warpPerspective(img, M, (img_cols, img_rows))
        # cv2.imshow('dst image', dst)
        # cv2.waitKey(0)

    except Exception as e:
        # This is the standard tmp exception print log.
        import sys
        logging.warning(
            f"Exception in {sys._getframe().f_code.co_name}, e:{e}")
        raise

    return dst


def fnAdd_saltNoise(img, n):
    """Add salt noise on image.

    :param img: image
    :type img: mat
    :param n: noise level, noise point number.
    :type n: int
    :return: updated image.
    :rtype: mat
    """

    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        # for gray image.
        if img.ndim == 2:
            img[j, i] = 255
        # for rgb image.
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255

    return img


def fnGet_augmented(img, distort_maxRatio, n=0):

    img = fnAdd_saltNoise(img, n)
    out = fnProjectiveTransform(img, distort_maxRatio)

    return out


def fnGet_dataset_dict(const_dataset_path):
    '''
    {
        "<category name>": [
            <image name>,
            <image name>
            ...
        ],
        ...
    }
    '''

    data_dict = {}

    folder_nameList = \
            [f for f in os.listdir(const_dataset_path)
             if os.path.isdir(os.path.join(const_dataset_path, f))]

    for folder_name in folder_nameList:

        folder_path = os.path.join(const_dataset_path, folder_name)
        image_pathList = \
            [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith("aug0_") and not f.startswith("aug1_")]

        data_dict[folder_name] = image_pathList

    return data_dict


def fnGet_detected_roiImage(img, each_result, larger_ratio=0.1):

    try:
        img_rows, img_cols = img.shape[:2]
        left, top, width, height = each_result['box']

        if larger_ratio > 0:
            pad_ratio = larger_ratio

            x_pad = int(img_cols * pad_ratio)
            y_pad = int(img_rows * pad_ratio)

            new_left = left - x_pad
            new_top = top - y_pad
            new_width = width + x_pad*2
            new_height = height + y_pad*2

            # Calibration
            if new_left + new_width > img_cols:
                new_width = img_cols - new_left

            if new_top + new_height > img_rows:
                new_height = img_rows - new_top

            if new_left < 0:
                new_left = 0

            if new_top < 0:
                new_top = 0

        else:
            new_left, new_top, new_width, new_height = \
                left, top, width, height

    except Exception as e:
        logging.warning("[warn] Exception...")
        new_left, new_top, new_width, new_height = \
            left, top, width, height

    out = img[new_top:new_top+new_height, new_left:new_left+new_width]
    # cv2.imshow('out', out)
    # cv2.waitKey(0)

    return out


def is_same_label(s, gt):

    if gt.find(s) < 0:
        return False

    print(f"\tFind {s} in {gt}")
    return True


def fnGenerate_augimg(frame, net, classes, gt):

    # Process image.
    detections = pre_process(frame, net)
    results = post_process(frame.copy(), detections, classes)

    img_list = []

    # Visulization by results.
    for each_result in results:

        print("\t\tbndbox: {}".format(each_result))

        class_id = each_result['class_id']
        confidence = each_result['confidence']

        class_name = classes[class_id]
        label1 = "\t\t{}: {:.2f}".format(class_name, confidence)
        print(label1)

        t, _ = net.getPerfProfile()
        label2 = '\t\tInference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        print(label2)

        if is_same_label(class_name, gt):

            # roi.
            roi = fnGet_detected_roiImage(frame.copy(), each_result, larger_ratio=0.1)
            img_list.append(roi)

            # augmented roi
            augmented = fnGet_augmented(roi, distort_maxRatio=0.2, n=200)
            img_list.append(augmented)

            # cv2.imshow('each roi', roi)
            # cv2.imshow('each roi augmented', augmented)
            # cv2.waitKey(0)

        else:
            print("\t\t[warn] {} should be: {}".format(class_name, gt))
            # cv2.imshow('ori image', frame)
            # cv2.waitKey(0)

    return img_list

