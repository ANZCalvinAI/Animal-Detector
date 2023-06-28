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
def detect(model_detector="yolov5", weight_detector, cls_custom=0.5, ci_custom, image):
    if model_detector != "yolov5":
        raise ValueError("only 'yolov5' is supported")
    detector = torch.hub.load("ultralytics/" + model_detector, "custom", path=weight_detector)
    out = detector(image)
    out = out.pandas().xyxy[0]
    out = out[out["name"] == cls_custom]
    if out.empty:
        raise ValueError(f"no detected {cls_custom} with confidence interval {ci_custom}.")
    else:
        if len(out) = 1:
            print(f"only 1 detected {cls_custom} with confidence interval {ci_custom}")
        else:
            out = out.loc[out["confidence"].idxmax()]
            print(
                f"2 or more detected {cls_custom} with condifence interval {ci_custom}.\n
                select the 1 {cls_custom} with the highest confidence interval."
            )
        xmin, ymin, xmax, ymax = out["xmin"], out["ymin"], out["xmax"], out["ymax"]
        image = cv2.imread(image)
        image = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    return image


# =======================================
# transform yolo output into effnet input
# =======================================
# def crop(image, out_detect):
#     return image[xyxy[0]:xyxy[1], xyxy[2]:xyxy[3]]


# =====================
# effnet classification
# =====================
def classify(model_classifier="effnet", weight_classifier, path_images):
    if model_classifier != "effnet":
        raise ValueError("only 'effnet' is supported")
    classifier = effnet(weight_detector)

# ...
