import numpy as np
import torch
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_v2_s as effnetv2


# ========
# detector
# ========
def detect(image, model_detector="yolov5x", weight_detector=None, cls_custom=None, ci_custom=0.5):
    # load image
    with open(image, "rb") as f:
        image = f.read()
        arr = np.asarray(bytearray(image), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)

    # check image
    if image is None:
        raise Exception(f"no valid image read via 'cv2.imread()'.\n")
    else:
        print("image loaded.\n")

    # check model
    if model_detector != "yolov5x":
        raise ValueError("only model 'yolov5x' supported for detector.\n")

    # load detector
    detector = torch.hub.load(
        f"ultralytics/{model_detector}",
        "custom",
        path=weight_detector,
        force_reload=True
    )
    print(f"specified weight '{weight_detector}' loaded.\n")

    # detect objects
    output = detector(image)
    output = output.pandas().xyxy[0]
    output = output[output["name"] == cls_custom]

    # check output
    if output.empty:
        raise Exception(f"no '{cls_custom}' detected with confidence interval >= '{ci_custom}'.")
    elif output.shape[0] == 1:
        print(f"1 '{cls_custom}' detected with confidence interval >= '{ci_custom}'.\n")
    else:
        output = output.loc[[output["confidence"].idxmax()]]
        ci_high = round(output["confidence"].iloc[0], 2)
        print(f"2 or more '{cls_custom}' detected with specified confidence interval >= {ci_custom}.\n")
        print(f"select 1 detected '{cls_custom}' with the higher confidence interval = {ci_high}.\n")

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

    return image


# ==========
# classifier
# ==========
def classify(image, model_classifier="effnetv2", weight_classifier=None, top_classes=5):
    # check model
    if model_classifier != "effnetv2":
        raise ValueError("only model 'effnetv2' is supported for classifier.")

    # check weight
    if weight_classifier is None:
        classifier = effnetv2(pretrained=True)

    # classify object
    with torch.no_grad():
        output = torch.nn.functional.softmax(classifier(image), dim=1)
    utils = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_convnets_processing_utils"
    )
    output = utils.pick_n_best(predictions=output, n=top_classes)

    return output


# =========
# inference
# =========
def inference(
    image,
    model_detector="yolov5x", weight_detector=None, cls_custom=None, ci_custom=0.5,
    model_classifier="effnetv2", weight_classifier=None, top_classes=5
):
    output = detect(
        image, model_detector=model_detector, weight_detector=weight_detector,
        cls_custom=cls_custom, ci_custom=ci_custom
    )

    output = classify(
        output, model_classifier=model_classifier, weight_classifier=weight_classifier,
        top_classes=top_classes
    )

    return output
