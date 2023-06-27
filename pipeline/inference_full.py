import os
import torch


def list_files(direct):
    r = []
    for root, dirs, files in os.walk(direct):
        for name in files:
            r.append(os.path.join(root, name))
    return r


# ==============
# yolo detection
# ==============
def detect(path_weight_yolo, path_images):
    yolov5 = torch.hub.load("ultralytics/yolov5", "custom", path=path_weight_yolo)
    images = list_files(path_images)
    out_detect = yolov5(images)
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
# def classify(...):
#     ...

# ...
