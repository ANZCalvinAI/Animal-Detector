import
from torch import no_grad
from torch.nn.functional import softmax
from PIL import Image
from utils import data_transform, get_weight_latest

# ====================
# customise parameters
# ====================
# project path parameter
path_project = "C:/Users/cz199/PycharmProjects/Animal-Detector/"

# =======================
# specify path parameters
# =======================
path_image = path_project + "datasets/iNat2021/images/"
path_weight = path_project + "resnet/weights/"

# =============
# set up device
# =============
# set up the device as GPU as first choice and CPU as alternative
device = device("cuda:0" if cuda.is_available() else "cpu")
print(f"device\n{device}\n")

# =======================
# set up model and weight
# =======================
"""
set up the model as ResNet 152.
use the most recently created weight in the weight path.
"""
else:
    # load the ResNet 152 model from local
    model = resnet152(weights=None)
    # search for the latest weight from local
    weight_latest = get_weight_latest(path_weight)
    # let state_dict load the latest weight from local
    state_dict = load(path_weight + weight_latest)
    # let the ResNet 152 model load the state_dict
    model.load_state_dict(state_dict)

# =================
# testing one image
# =================
input_image = Image.open(filename)
preprocess = data_transform["test"]

input_tensor = preprocess(input_image)
# create a mini batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

with no_grad():
    output = model(input_batch)
    
probabilities = softmax(output[0], dim=0)

# =====================
# calculate top 5 error
# =====================
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
