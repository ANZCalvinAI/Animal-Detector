Algorithm: YOLO-ResNet Pipeline (e.g. for Insects)
Input:
    image: Data,
    insect_class_index: Int,
    yolov5: Model with Fine Tuned Weights,
    resnet: Model with Fine Tuned Weights
Output:
    ResNet Predicted Class

1. # detect objects from an image.
   `out_detect = yolov5(image)`;

# by the way, out_detect should be like:
# [
#     [1_class_index, 1_x_centre, 1_y_centre, 1_width, 1_height],  # for detected object 1
#     ...,
#     [N_class_index, N_x_centre, N_y_centre, N_width, N_height]   # for detected object N
# ]

2. # find the insect object from the detected object(s).
   for object in out_detect:
       if object[1] == insect_class_index:
           return object

3. # input the insect object to the ResNet model, classifying it.
   out_classify = resnet(object).

4. # return the ResNet classification result.
   return out_classify