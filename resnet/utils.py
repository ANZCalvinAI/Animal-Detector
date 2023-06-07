from os import listdir
import numpy as np
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize,\
    CenterCrop

"""
define a standard data transform format
(a) training:
RandomResizedCrop: ...;
Normalize: ....

(b) validation:
Resize: ...;
CenterCrop: ...;
Normalize: ....
"""
data_transform = {
    "train": Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    "val": Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}


"""
define function get_weight_latest(). 
this function outputs the filename of the latest weight, given the path of weights as the input
"""
def get_weight_latest(path_weight):
    time_list = np.array([])
    for filename in listdir(path_weight):
        time = int(filename[10:24])
        time_list = np.append(time_list, time)
    out = str(int(np.amax(time_list)))
    out = "resnet-" + out + ".pth"
    return out
