import os
import numpy as np

path_test = 'C:\Users\cz199\PycharmProjects\yolov5_classify\datasets\iNat2017\labels\test\'
path_pred = '...'
n_classes = 13

# ================
# read test labels
# ================
# initialise a y_test list as an empty list 
y_test = np.array([])

for file_name in os.listdir(path_test):
    # for every label file in the test path
    with open(path_test + file_name, 'r') as file:
        # take the first two elements of the label file as the classification result, e.g. string(1, 3) = class 13, string(1, space) = class 1.
        cls = file.read()[0:2]
        # transform the classification result from the string into an integer, e.g. string(1, 3) = int(13), string(1, space) = int(1).
        cls = int(cls)
        # append the classification result into the y_test list.
        y_test = np.append(y_test, cls)
        
# =====================
# read predicted labels
# =====================
# initialise a y predicted list as an empty list 
y_pred = np.array([])

for file_name in os.listdir(path_pred):
    # for every label file in the prediction path
    with open(path_pred + file_name, 'r') as file:
        # take the first two elements of the label file as the classification result, e.g. string(1, 3) = class 13, string(1, space) = class 1.
        cls = file.read()[0:2]
        # transform the classification result from the string into an integer, e.g. string(1, 3) = int(13), string(1, space) = int(1).
        cls = int(cls)
        # append the classification result into the y_pred list.
        y_pred = np.append(y_pred, cls)

# ==============================
# compare y_test and y_predicted
# ==============================
'''
comparative tuple
out = [
    [sample 1 test, sample 1 pred],
    ...
    [sample N test, sample N pred]
]
'''
out = np.array([
    y_test,
    y_predicted
])
out = np.transpose(out)

# respective comparison by class (predicted class, rather than true class)
for k in range(n_classes):
    out_k = out[out[:, 1] == k]
    # total predicted positive = false positive + true positive
    predicted_positive = len(out_k)
    # true positive
    true_positive = np.sum(out_k[:, 0] == out_k[:, 1])
    # precision = true positive / predicted positive
    precision = true_positive / predicted_positive
    print(f'class {k}')
    print(f'precision {precision}')
  
