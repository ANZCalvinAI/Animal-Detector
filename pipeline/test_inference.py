import os
from inference import infer

# configure paths
path_pipeline = os.path.abspath(".")
path_weight = os.path.join(path_pipeline, "weights")
weight_detector = os.path.join(path_weight, "yolov5x.pt")
weight_classifier = os.path.join(path_weight, "effnetv2m.pth")

path_project = os.path.abspath("..")
path_dataset = os.path.join(path_project, "datasets\pipeline")

# load image
image = os.path.join(path_dataset, "image1.jpg")

output = infer(
    image=image, cls_custom="Insecta",
    weight_detector=weight_detector, weight_classifier=weight_classifier
)
output = [cls[0] for cls in output[0]]
print(f"output\n{output}")
