# Animal-Detector

![alt text](https://github.com/ANZCalvinAI/Animal-Detector/blob/main/Koalas.jpg?raw=true)

## 0. Architecture
Fine tuned YOLOv5 Large + fine tuned ResNet 152

## 1. YOLO
### 1.1 Datasets
- Dataset: iNaturalist 2017
- Dataset Path: "datasets/iNat2017/"
- Data Format Path: "yolo/data/inat2017.yaml" 

### 1.2 Models
- Model Type: YOLOv5 Large
- Model Path: "yolo/models/yolov5l.yaml"
- Hyperparameter Configs: Scratch High
- Hyperparameter Configs Path: "data/hyps/hyp.scratch-high.yaml"

### Document
> Jocher G, Chaurasia A, Stoken A, Borovec J, Kwon Y, Michael K, Fang J, Yifu Z, Wong C, Montes D, Wang Z. Ultralytics/yolov5: v7. 0-YOLOv5 SotA realtime instance segmentation. Zenodo. 2022 Nov.

## 2. ResNet
### 2.1 Datasets
- Dataset: iNaturalist 2021
- Dataset Path: "datasets/iNat2021/"

### 2.2 Models
- Model Type: ResNet 152
- Model Path: "resnet/model.py"

### 2.3 Weights
- Weight Path: "resnet/weights/"
- Weight Filename Format: "resnet152-YYYYMMDDHHMMSS.pth". For example, the filename of a weight would be "resnet-20230101000000.pth" for a weight created at 2023-01-01 00:00:00.
- Pretrained Weight Filename: "resnet152-19000101000000.pth"

### 2.4 Training
- Script Path: "resnet/train.py"
- Training Logics: Train the most recently created weight, then save it as a more recently created weight.

### Document
> He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).

## References
- Jocher G, Chaurasia A, Stoken A, Borovec J, Kwon Y, Michael K, Fang J, Yifu Z, Wong C, Montes D, Wang Z. Ultralytics/yolov5: v7. 0-YOLOv5 SotA realtime instance segmentation. Zenodo. 2022 Nov.
- He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
