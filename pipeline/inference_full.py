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
def detect(model_detector="yolov5", weight_detector, cls_custom, ci_custom=0.5, image):
    if model_detector != "yolov5":
        raise ValueError("only 'yolov5' is supported for detector.")
    detector = torch.hub.load("ultralytics/" + model_detector, "custom", path=weight_detector)
    out = detector(image)
    out = out.pandas().xyxy[0]
    out = out[out["name"] == cls_custom]
    
    # case 1: nothing detected
    if out.empty:
        raise ValueError(f"no detected {cls_custom} with confidence interval {ci_custom}.")
    else:
        # case 2: 1 detected
        if len(out) = 1:
            print(f"only 1 detected {cls_custom} with confidence interval {ci_custom}.")
        # case 3: 2 or more detected. select the 1 with highest confidence interval
        else:
            out = out.loc[out["confidence"].idxmax()]
            print(
                f"2 or more detected {cls_custom} with condifence interval {ci_custom}.\n
                select the 1 {cls_custom} with the highest confidence interval."
            )
        # get the bounding box and cut the image based on the acquired bounding box
        xmin, ymin, xmax, ymax = out["xmin"], out["ymin"], out["xmax"], out["ymax"]
        image = cv2.imread(image)
        image_cropped = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    return cls_custom, image_cropped


# =====================
# effnet classification
# =====================
# def classify(model_classifier="effnet", weight_classifier, out_detect):
#     if model_classifier != "effnet":
#         raise ValueError("only 'effnet' is supported for classifier.")
#     classifier = effnet(weight_detector)
    
# ...
