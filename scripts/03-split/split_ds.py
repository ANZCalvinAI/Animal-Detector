import json
import os
import logging
import pprint
import shutil
from pathlib import Path

src_dirPath = "/home/ubuntu/dpinsw/classification-resnet/Animal-Detector/datasets/iNat2021/"
dst_dirPath = "./split_output"

# key: supercategory, name
category_dict = {}

# key: name
img_dict = {}


def create_newFolder(dirPath, force=True):
    """Create one folder.

    :param dirPath: new folder path
    :type dirPath: str
    :param force: force or not, defaults to True
    :type force: bool, optional
    :return: status
    :rtype: bool
    """
    logging.debug("In create_newFolder()\n")

    if Path(dirPath).exists() is True:
        if force is True:
            logging.debug("Delete the content in it: {}".format(dirPath))
            logging.debug("Old: {} will be deleted".format(dirPath))
            shutil.rmtree(dirPath)
            logging.debug("New: {} will be created.".format(dirPath))
            os.makedirs(dirPath)
        else:
            # Cannot remove the old one, tell the user.
            logging.warning("{} exists, I will not create it.".format(dirPath))
            return False
    else:
        os.makedirs(dirPath, exist_ok=True)

    if Path(dirPath).exists is False:
        logging.error("fail to create folder: {}".format(dirPath))
        return False

    return True


def fnMoveImgToFolder(img_path, img_dirPath):
    shutil.move(img_path, img_dirPath)


def fnCreateLabelFile(label_filepath, category_id):

    # Create a new file
    # Add one line
    f = open(label_filepath, "w")
    f.write("{}".format(category_id))
    f.close()


def load_dirNameList(dirPath):

    onlydirs = \
        [f for f in os.listdir(dirPath)
            if os.path.isdir(os.path.join(dirPath, f))]

    return onlydirs


def load_fileNameList(dirPath):

    onlyfiles = \
        [f for f in os.listdir(dirPath)
            if os.path.isfile(os.path.join(dirPath, f))]

    return onlyfiles


def filter_dirNameList(dirNameList, kw_list):
    
    all_filtered = []

    for kw in kw_list:
        filtered = [x for x in dirNameList if kw in x]
        all_filtered = all_filtered + filtered

    all_filtered.sort()
    return all_filtered



def get_statistics_of_species(species_dirPath):

    dirNameList = load_dirNameList(species_dirPath)
    print(dirNameList)
 
    species_statistic_list = [{}, {}, {}, {}, {}, {}, {}]

    for dir_idx, each_dirName in enumerate(dirNameList):
        
        section_list = each_dirName.split('_')
        if len(section_list) != 8:
            print("[err] len(section_list) is {}".format(len(section_list)))
            exit(1)
        
        category_id = section_list[0]

        jie   = section_list[1]
        men   = section_list[2]
        gang  = section_list[3]
        mu    = section_list[4]
        ke    = section_list[5]
        shu   = section_list[6]
        zhong = section_list[7]


        for idx, sec_name in enumerate([jie, men, gang, mu, ke, shu, zhong]):

            if sec_name not in species_statistic_list[idx].keys():
                species_statistic_list[idx][sec_name] = 1
            else:
                species_statistic_list[idx][sec_name] += 1

    print(len(species_statistic_list[0]), species_statistic_list[0])
    print(len(species_statistic_list[1]), species_statistic_list[1])
    print(len(species_statistic_list[2]), species_statistic_list[2])
    print(len(species_statistic_list[3]), species_statistic_list[3])
    print(len(species_statistic_list[4]), species_statistic_list[4])
    print(len(species_statistic_list[5]), species_statistic_list[5])
    print(len(species_statistic_list[6]), species_statistic_list[6])

    return None


def get_tree_statistics(species_dirPath):

    species_tree_dict = {}

    dirNameList = load_dirNameList(species_dirPath)
 
    for dir_idx, each_dirName in enumerate(dirNameList):
        
        section_list = each_dirName.split('_')
        if len(section_list) != 8:
            print("[err] len(section_list) is {}".format(len(section_list)))
            exit(1)
        
        category_id = section_list[0]

        jie   = section_list[1]
        men   = section_list[2]
        gang  = section_list[3]
        mu    = section_list[4]
        ke    = section_list[5]
        shu   = section_list[6]
        zhong = section_list[7]

        num = 'num'

        if jie not in species_tree_dict.keys():
            species_tree_dict[jie] = dict(num=1)
        else:
            species_tree_dict[jie][num] += 1

        if men not in species_tree_dict[jie].keys():
            species_tree_dict[jie][men] = dict(num=1)
        else:
            species_tree_dict[jie][men][num] += 1

        if gang not in species_tree_dict[jie][men].keys():
            species_tree_dict[jie][men][gang] = dict(num=1)
        else:
            species_tree_dict[jie][men][gang][num] += 1

        if mu not in species_tree_dict[jie][men][gang].keys():
            species_tree_dict[jie][men][gang][mu] = dict(num=1)
        else:
            species_tree_dict[jie][men][gang][mu][num] += 1

        if ke not in species_tree_dict[jie][men][gang][mu].keys():
            species_tree_dict[jie][men][gang][mu][ke] = dict(num=1)
        else:
            species_tree_dict[jie][men][gang][mu][ke][num] += 1

        # if shu not in species_tree_dict[jie][men][gang][mu][ke].keys():
        #     species_tree_dict[jie][men][gang][mu][ke][shu] = dict(num=1)
        # else:
        #     species_tree_dict[jie][men][gang][mu][ke][shu][num] += 1

        # if zhong not in species_tree_dict[jie][men][gang][mu][ke][shu].keys():
        #     species_tree_dict[jie][men][gang][mu][ke][shu][zhong] = dict(num=1)
        # else:
        #     species_tree_dict[jie][men][gang][mu][ke][shu][zhong][num] += 1

    pprint.pprint(species_tree_dict)

    return None


def fnSplit_ds(kw_list, src_images_dirPath,  src_labels_dirPath,  dst_images_dirPath,  dst_labels_dirPath):

    dirNameList = load_dirNameList(src_images_dirPath)
    filteredNameList = filter_dirNameList(dirNameList, kw_list)

    create_newFolder(dst_images_dirPath, force=True)
    create_newFolder(dst_labels_dirPath, force=True)

    for label_id, species_dirName in enumerate(filteredNameList):

        dst_photo_species_dirPath = os.path.join(dst_images_dirPath, species_dirName)
        dst_label_species_dirPath = os.path.join(dst_labels_dirPath, species_dirName)

        create_newFolder(dst_photo_species_dirPath, force=True)
        create_newFolder(dst_label_species_dirPath, force=True)

        # Start to process each species.
        species_dirPath = os.path.join(src_images_dirPath, species_dirName)
        photo_names = load_fileNameList(species_dirPath)
        
        # Start to copy/move each photo.
        for photo_name in photo_names:

            # ---------------------------
            # detection for optimization.
            # ---------------------------

            # Create dst photo
            src_photo_path = os.path.join(src_images_dirPath, species_dirName, photo_name)
            dst_photo_path = os.path.join(dst_images_dirPath, species_dirName, photo_name)
            
            print(src_photo_path)
            print(dst_photo_path)
            fnMoveImgToFolder(src_photo_path, dst_photo_path)

            # Create dst label.
            short_name, _ = os.path.splitext(photo_name)
            label_file_name = "{}.txt".format(short_name)

            src_label_path = os.path.join(src_labels_dirPath, species_dirName, label_file_name)
            dst_label_path = os.path.join(dst_labels_dirPath, species_dirName, label_file_name)

            print(src_label_path)
            print(dst_label_path)
            fnMoveImgToFolder(src_label_path, dst_label_path)


def main(kw_list):

    #
    # 01. get statistics in ./train
    #
    src_images_train_dirPath = os.path.join(src_dirPath, "images/train")
    src_images_val_dirPath = os.path.join(src_dirPath, "images/val")
    src_images_test_dirPath = os.path.join(src_dirPath, "images/test")

    src_labels_train_dirPath = os.path.join(src_dirPath, "labels/train")
    src_labels_val_dirPath = os.path.join(src_dirPath, "labels/val")
    src_labels_test_dirPath = os.path.join(src_dirPath, "labels/test")

    dst_images_train_dirPath = os.path.join(dst_dirPath, "images/train")
    dst_images_val_dirPath = os.path.join(dst_dirPath, "images/val")
    dst_images_test_dirPath = os.path.join(dst_dirPath, "images/test")

    dst_labels_train_dirPath = os.path.join(dst_dirPath, "labels/train")
    dst_labels_val_dirPath = os.path.join(dst_dirPath, "labels/val")
    dst_labels_test_dirPath = os.path.join(dst_dirPath, "labels/test")


    # get_statistics_of_species(src_images_train_dirPath)
    get_tree_statistics(src_images_train_dirPath)

    # return None


    #
    # 02. do splitting.
    #
    print(kw_list)

    fnSplit_ds(kw_list, src_images_train_dirPath, src_labels_train_dirPath, dst_images_train_dirPath, dst_labels_train_dirPath)
    fnSplit_ds(kw_list, src_images_val_dirPath,   src_labels_val_dirPath,   dst_images_val_dirPath,   dst_labels_val_dirPath)
    fnSplit_ds(kw_list, src_images_test_dirPath,  src_labels_test_dirPath,  dst_images_test_dirPath,  dst_labels_test_dirPath)

    return None


if __name__ == '__main__':

    if not create_newFolder(dst_dirPath, force=False):
        print("Please check ./split_output, and create a new one.")
        exit()
    else:
        create_newFolder(f"{dst_dirPath}/images", force=True)
        create_newFolder(f"{dst_dirPath}/images/train", force=True)
        create_newFolder(f"{dst_dirPath}/images/val", force=True)
        create_newFolder(f"{dst_dirPath}/images/test", force=True)
        create_newFolder(f"{dst_dirPath}/labels", force=True)
        create_newFolder(f"{dst_dirPath}/labels/train", force=True)
        create_newFolder(f"{dst_dirPath}/labels/val", force=True)
        create_newFolder(f"{dst_dirPath}/labels/test", force=True)


    kw_dict = {'Hymenoptera': 177, 'Orthoptera': 98, 'Hemiptera': 164, 'Coleoptera': 236, 'Diptera': 85, 'Mantodea': 14, 'Neuroptera': 7, 'Megaloptera': 3, 'Blattodea': 8, 'Zygentoma': 3, 'Phasmida': 4, 'Dermaptera': 2, 'Mecoptera': 2, 'Ephemeroptera': 1, 'Psocodea': 1}

    kw_list = []
    category_count = 0
    for k, v in kw_dict.items():
        kw_list.append(k)
        category_count += v

    print("Category Count: {}".format(category_count))

    # Will split ds under this kw category.
    main(kw_list)

