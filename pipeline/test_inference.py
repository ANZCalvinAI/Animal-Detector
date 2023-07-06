import os
from inference import batch_infer

# configure path
path_pipeline = os.path.abspath(".")
path_weight = os.path.join(path_pipeline, "weights")
weight_detector = os.path.join(path_weight, "yolov5x.pt")
weight_classifier = os.path.join(path_weight, "effnetv2m.pth")

path_project = os.path.abspath("..")
path_dataset = os.path.join(path_project, "datasets\pipeline")

# load image
image1 = os.path.join(path_dataset, "image1.jpg")
image2 = os.path.join(path_dataset, "image2.jpeg")

output = batch_infer(
    images=[image1, image2], cls_custom="Insecta",
    weight_detector=weight_detector, weight_classifier=weight_classifier
)
print(output)
