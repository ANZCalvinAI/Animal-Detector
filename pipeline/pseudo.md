### Input:
```
image: data;
insect_class_index: int;
detect: a detection function, needs specifing (a) model and weight and (b) data in parameters;
classify: a classification function, needs specifing (a) model and weight and (b) data in parameters.
```

### Output:
```
out_classify: int
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
