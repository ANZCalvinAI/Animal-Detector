import os
from inference import inference

# configure paths
path_pipeline = os.path.abspath(".")
"""
path_weight = os.path.join(path_pipeline, "weights")
weight_detector = os.path.join(path_weight, "yolov5x.pt")
"""
path_asset = os.path.join(path_pipeline, "assets")
weight_detector = os.path.join(path_asset, "yolov5x.pt")
weight_classifier = None  # os.path.join(path_asset, "effnetv2.pt/.pth")
utils_classifier = None

path_project = os.path.abspath("..")
path_dataset = os.path.join(path_project, "datasets\pipeline")

# load image
image1 = os.path.join(path_dataset, "image1.jpg")

inference(image=image1, weight_detector=weight_detector, cls_custom="Insecta")
