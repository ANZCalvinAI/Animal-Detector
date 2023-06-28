### Input:
```
image;
cls_preset: str;   # class to detect, not defaulted
ci_preset: float;  # confidence interval level to recognise a detection, defaulted to 0.5
model_detector;    # detector model, defaulted to ...
weight_detector;   # detector weight, defaulted to ...
model_classifier;  # classifer model, defaulted to ...
weight_classifer:  # classifer weight, defaulted to ...
```

### Output:
```
out_classify: str
```

### Functions Defined in the Script pipeline.py
```
1. detect()
"""a detection function"""
inputs:
- (a) model_detector,
- (b) weight_detector,
- (c) cls_preset,
- (d) ci_preset,
- (e) image.
output:
- out_detect = [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]  # bounding box

2. crop()
"""a cropping function"""
inputs:
- (a) image;
- (b) out_detect.
outputs:
- image_cropped

3. classify()
"""a classification function"""
inputs:
- (a) model_classifier,
- (b) weight_classifer,
- (c) image_cropped.
output:
- out_classify
```

## Algorithm: YOLO-ResNet Pipeline (e.g. for Insects)
### 1. detect objects from an image.
```
out_detect = detect(image)
```

by the way, out_detect should be an object list:
```
[
    [1_class_index, 1_x_centre, 1_y_centre, 1_width, 1_height],  # for detected object 1
    ...,
    [N_class_index, N_x_centre, N_y_centre, N_width, N_height]   # for detected object N
]
```

### 2. find the insect object from out_detect.
```
for object in object_list:
    if object[1] == insect_class_index:
        return object
```

### 3. input the insect object to the ResNet model, classifying it.
```
out_classify = classify(object)
```

### 4. return the ResNet classification result.
```
return out_classify
```
