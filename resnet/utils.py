from os import listdir
import numpy as np


# define function get_weight_latest(). 
# this function outputs the filename of the latest weight, given the path of weights as the input
def get_weight_latest(path_weight):
    time_list = np.array([])
    for filename in listdir(path_weight):
        time = int(filename[7:21])
        time_list = np.append(time_list, time)
    out = str(int(np.amax(time_list)))
    out = "resnet-" + out + ".pth"
    return out
