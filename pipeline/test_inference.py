import os
from inference import inference

# ============
# config paths
# ============
path_project = os.path.abspath("..")
path_pipeline = os.path.abspath(".")

path_weight = path_pipeline + "/weights_fine_tuned"
path_weight_yolo = path_weight + "/yolov5x.pt"

path_images = path_project + "/datasets/pipeline/images"

# =========
# inference
# =========

path_weight_yolo = "weights_fine_tuned/yolov5x.pt"
path_images = "./dataset"

inference(path_weight_yolo=path_weight_yolo, path_images=path_images)
