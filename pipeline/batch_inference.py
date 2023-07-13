import numpy as np
import torch
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_v2_m as effnetv2m


# ======
# device
# ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============
# batch inference
# ===============
def batch_infer(
    images, cls_custom=None, ci_custom=0.1, top_classes=5,
    model_detector="yolov5x", weight_detector=None,
    model_classifier="effnetv2m", weight_classifier=None
):
    # check image quantity
    if len(images) != 2:
        raise ValueError("only image quantity '2' supported for batch inference.")
  
    # check models
    if model_detector != "yolov5x":
        raise ValueError("only model 'yolov5x' supported for detector.\n")

    if model_classifier != "effnetv2m":
        raise ValueError("only model 'effnetv2m' is supported for classifier.")
    
    # load models
    detector = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=weight_detector,
        force_reload=True
    )
    detector = detector.to(device)

    classifier = effnetv2m()
    classifier.load_state_dict(torch.load(weight_classifier, map_location=device))

    # detect and classify objects
    outputs = []
  
    for image in images:
        # load image
        with open(image, "rb") as f:
            image = f.read()
            arr = np.asarray(bytearray(image), dtype=np.uint8)
            image = cv2.imdecode(arr, -1)

        # check image
        if image is None:
            raise Exception(f"no valid image read via 'cv2.imread()'.\n")

        # detect objects
        output = detector(image)
        output = output.pandas().xyxy[0]
        output = output[output["name"] == cls_custom]

        # check output
        if output.empty:
            raise Exception(f"no '{cls_custom}' detected with confidence interval >= '{ci_custom}'.")
        else:
            if output.shape[0] >= 2:
                output = output.loc[[output["confidence"].idxmax()]]

        # calculate bounding box
        xmin, xmax, ymin, ymax = output["xmin"], output["xmax"], output["ymin"], output["ymax"]

        # crop image
        image = image[int(ymin):int(ymax), int(xmin):int(xmax)]

        # configure transformation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # transform image
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = torch.nn.functional.softmax(classifier(image), dim=1)
        utils = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_convnets_processing_utils"
        )
        output = utils.pick_n_best(predictions=output, n=top_classes)
        output = [cls[0] for cls in output[0]]

        outputs = outputs.append(output)

    output = [cls for cls in outputs[0] if cls in outputs[1]]
  
    return output
