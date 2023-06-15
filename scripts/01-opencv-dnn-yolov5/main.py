import cv2
import numpy as np
from yolov5 import (
    FONT_FACE, FONT_SCALE, THICKNESS,
    BLACK, BLUE, YELLOW, RED,
    pre_process,
    post_process,
)


if __name__ == '__main__':
    '''
    Prepare ./models before running.
    models
    ├── labels.txt
    └── yolov5l.onnx

    '''

	const_classesFile = "models/labels.txt"
	const_modelWeights = "models/yolov5l.onnx"
	const_test_img_path = "imgs/1.jpg"

	# Load class names.
	classes = None
	with open(const_classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Load image.
	frame = cv2.imread(const_test_img_path)

	# Give the weight files to the model and load the network using them.
	net = cv2.dnn.readNet(const_modelWeights)

	# Process image.
	detections = pre_process(frame, net)
	img = post_process(frame.copy(), detections, classes)

	# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	t, _ = net.getPerfProfile()
	label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
	cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

	cv2.imshow('Output', img)
	cv2.waitKey(0)

