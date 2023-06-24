import os
import torch

# ============
# config paths
# ============
path_project = os.path.abspath("..")
path_pipeline = os.path.abspath(".")

path_weight = path_pipeline + "/weights_fine_tuned"
path_weight_yolo = path_weight + "/yolov5x.pt"

path_images = path_project + "/datasets/pipeline/images"

# =========
# detection
# =========
yolov5 = torch.hub.load("ultralytics/yolov5", "custom", path=path_weight_yolo)


# detection inference
def list_files(direct):
    r = []
    for root, dirs, files in os.walk(direct):
        for name in files:
            r.append(os.path.join(root, name))
    return r


images = list_files(path_images)

print(images)

out_detect = yolov5(images)

out_detect.print()
out_detect.save()
