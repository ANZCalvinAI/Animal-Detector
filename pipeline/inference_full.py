import os
import torch
from torchvision.models import efficientnet_v2_s as effnetv2


def list_files(direct):
    r = []
    for root, dirs, files in os.walk(direct):
        for name in files:
            r.append(os.path.join(root, name))
    return r


# ==============
# yolo detection
# ==============
def detect(model_detector="yolov5", weight_detector, path_images):
    detector = torch.hub.load("ultralytics/" + model_detector, "custom", path=weight_detector)
    images = list_files(path_images)
    out_detect = detector(images)
    with open("output.txt", "w") as file:
        for i, image in enumerate(images):
            out = out_detect.xyxy[i]
            for *xyxy, conf, cls in out:
                file.write(f"{i} {out_detect.names[int(cls)]} {conf} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]}\n")

# =======================================
# transform yolo output into effnet input
# =======================================
...

# =====================
# effnet classification
# =====================
def classify(model_classifier="effnetv2", weight_classifier, path_images):
    if model_classifier="effnetv2":
        classifier = effnetv2(weight_detector)

# ...
