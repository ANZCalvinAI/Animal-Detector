# Animal-Detector

![alt text](https://github.com/ANZCalvinAI/Animal-Detector/blob/main/Koalas.jpg?raw=true)

## 0. Architecture
Fine tuned YOLOv5 Large + fine tuned ResNet 152

## 1. YOLO
### Document
> Jocher G, Chaurasia A, Stoken A, Borovec J, Kwon Y, Michael K, Fang J, Yifu Z, Wong C, Montes D, Wang Z. Ultralytics/yolov5: v7. 0-YOLOv5 SotA realtime instance segmentation. Zenodo. 2022 Nov.

## 2. ResNet
### 2.1 Datasets
- Dataset Path: datasets/iNat2021/

### 2.2 Models
- Model: ResNet 152
- Model Path: resnet/model.py

### 2.3 Weights
- Weight Path: resnet/weights/  
- Weight Filename Format: "resnet-YYYYMMDDHHMMSS.pth". For example, the filename of a weight would be "resnet-20230101000000.pth" for a weight created at 2023-01-01 00:00:00.
- Pretrained Weight Filename: "resnet-19000101000000.pth"

### 2.4 Training
- Script Path: resnet/train.py

### Document
> He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).

## References
- Jocher G, Chaurasia A, Stoken A, Borovec J, Kwon Y, Michael K, Fang J, Yifu Z, Wong C, Montes D, Wang Z. Ultralytics/yolov5: v7. 0-YOLOv5 SotA realtime instance segmentation. Zenodo. 2022 Nov.
- He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
