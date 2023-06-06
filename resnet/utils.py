from os import listdir
import numpy as np


def get_weight_latest(path_weight):
    time_list = np.array([])
    for filename in listdir(path_weight):
        time = int(filename[7:21])
        time_list = np.append(time_list, time)
    out = str(int(np.amax(time_list)))
    out = "resnet-" + out + ".pth"
    return out
