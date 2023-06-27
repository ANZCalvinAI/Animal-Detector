import os
import torch
from torchvision.models import efficientnet_v2_s as effnet


def list_files(direct):
    r = []
    for root, dirs, files in os.walk(direct):
        for name in files:
            r.append(os.path.join(root, name))
    return r


# ==============
# yolo detection
# ==============
def detect(model_detector="yolov5", weight_detector, cls_preset, ci_preset, image):
    detector = torch.hub.load("ultralytics/" + model_detector, "custom", path=weight_detector)
    out = detector(image)
    for *xyxy, conf, cls in out:
        if cls == cls_preset:
            if conf >= ci_preset:
                box = [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
                # ...
            eles:
                print(f"{cls_preset} detected, but confidence level < {ci_preset}.")
        else:
            print(f"{cls_preset} not detected.")

# =======================================
# transform yolo output into effnet input
# =======================================
...

# =====================
# effnet classification
# =====================
def classify(model_classifier="effnet", weight_classifier, path_images):
    if model_classifier="effnet":
        classifier = effnet(weight_detector)

# ...
