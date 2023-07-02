import json
import os
import logging
import shutil
from pathlib import Path

src_dirPath = "output/test"

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


def fnCreateLabelFile(label_filepath, category_id):

    # Create a new file
    # Add one line
    f = open(label_filepath, "w")
    f.write("{}".format(category_id))
    f.close()


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


def filter_dirNameList(dirNameList, kw):
    
    filtered = \
        [x for x in dirNameList
            if kw in x]
    
    filtered.sort()
    return filtered


def main(wk):

    # 01, load all labels.
    dirNameList = load_dirNameList(src_dirPath)

    # 02, filter out and find labels of for this model.
    filteredNameList = filter_dirNameList(dirNameList, kw)

    for label_id, species_dirName in enumerate(filteredNameList):

        # Start to process each species.
        species_dirPath = os.path.join(src_dirPath, species_dirName)
        photo_names = load_fileNameList(species_dirPath)
        
        print("{} {} {}".format(label_id, species_dirName, len(photo_names)))

        # Create dst dir.
        dst_images_dirPath = os.path.join("imgs", species_dirName)
        dst_labels_dirPath = os.path.join("labels", species_dirName)

        if not create_newFolder(dst_images_dirPath, force=False):
            print("[error] Cannot be here: 01")
            exit()
 
        if not create_newFolder(dst_labels_dirPath, force=False):
            print("[error] Cannot be here: 02")
            exit()


        # Start to process each photo.
        for photo_name in photo_names:

            # ---------------------------
            # detection for optimization.
            # ---------------------------

            # Create dst photo
            src_photo_path = os.path.join(species_dirPath, photo_name)
            dst_photo_path = os.path.join(dst_images_dirPath, photo_name)
            
            print(src_photo_path)
            print(dst_photo_path)
            fnMoveImgToFolder(src_photo_path, dst_photo_path)

            # Create dst label.
            short_name, _ = os.path.splitext(photo_name)
            label_file_name = "{}.txt".format(short_name)
            dst_label_path = os.path.join(dst_labels_dirPath, label_file_name)

            fnCreateLabelFile(dst_label_path, label_id)

        # Done.


if __name__ == '__main__':

    if not create_newFolder("imgs", force=False):
        print("Please check ./imgs")
        exit()

    # kw = "Aves_"
    kw = "Insecta"
    main(kw)

