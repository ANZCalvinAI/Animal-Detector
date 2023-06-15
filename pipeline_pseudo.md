### Input:
```
image: data,  
insect_class_index: int,  
yolov5: model with fine tuned weights,  
resnet: model with fine tuned weights  
```

### Output:
```
out_classify: int
```

## Algorithm: YOLO-ResNet Pipeline (e.g. for Insects)
### 1. detect objects from an image.
```
object_list = yolov5(image)
```

by the way, object_list should be like:
```
[
    [1_class_index, 1_x_centre, 1_y_centre, 1_width, 1_height],  # for detected object 1
    ...,
    [N_class_index, N_x_centre, N_y_centre, N_width, N_height]   # for detected object N
]
```

### 2. find the insect object from object_list.
```
for object in object_list:
    if object[1] == insect_class_index:
        return object
```

3. input the insect object to the ResNet model, classifying it.
```
out_classify = resnet(object)
```

4. return the ResNet classification result.
```
return out_classify
```
