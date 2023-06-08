import os
from json import dumps
from torch import device, cuda, load, save
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.hub import load as hub_load
from model import resnet152
from torchvision.datasets import ImageFolder
from utils import data_transform, get_weight_latest
from datetime import datetime

# ====================
# customise parameters
# ====================
# project path parameter
path_project = "C:/Users/cz199/PycharmProjects/Animal-Detector/"

# training parameters
params_training = {
    "batch_size": 16,  # training batch size
    "epochs_max": 3    # training maximal epochs
}

# =======================
# specify path parameters
# =======================
path_image = path_project + "datasets/iNat2021/images/"
path_weight = path_project + "resnet/weights/"

# =====================
# load and process data
# =====================
train_dataset = ImageFolder(
    root=path_image + "train",
    transform=data_transform["train"]
)
train_num = len(train_dataset)
print(f"training set sample size\n{train_num}\n")

animal_list = train_dataset.class_to_idx
print(f"animal classes\n{animal_list}\n")
cla_dict = dict((val, key) for key, val in animal_list.items())
print(f"class dictionary\n{cla_dict}\n")
json_str = dumps(cla_dict, indent=4)
with open("class_indices.json", "w") as json_file:
    json_file.write(json_str)

train_loader = DataLoader(
    train_dataset,
    batch_size=params_training["batch_size"],
    shuffle=True,
    num_workers=0
)

val_dataset = ImageFolder(
    root=path_image + "val",
    transform=data_transform["val"]
)
val_num = len(val_dataset)
validate_loader = DataLoader(
    val_dataset,
    batch_size=params_training["batch_size"],
    shuffle=False,
    num_workers=0
)

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
download and use the pretrained weight if there is no existing weight in the weight path;
otherwise use the most recently created weight in the weight path.
"""
# if the weight folder has not been created
if not os.path.exists(path_weight):
    # load the model and the pretrained weight from Torch Hub
    model = hub_load("pytorch/vision:v0.10.0", "resnet152", weights=None)
# if there is a weight folder
else:
    # if there is no weight file in the weight folder
    if not len(os.listdir(path_weight)):
        # load the model and the pretrained weight from Torch Hub
        model = hub_load("pytorch/vision:v0.10.0", "resnet152", weights=None)
    # if there is one or more weight files in the weight folder
    else:
        # load the ResNet 152 model from local
        model = resnet152(weights=None)
        # search for the latest weight from local
        weight_latest = get_weight_latest(path_weight)
        # let state_dict load the latest weight from local
        state_dict = load(path_weight + weight_latest)
        # let the ResNet 152 model load the state_dict
        model.load_state_dict(state_dict)

print(f"ResNet model architecture\n{model}\n")

model.to(device)
in_channel = model.fc.in_features
model.fc = Linear(in_channel, 5)

# ========
# training
# ========
# set up the loss as the cross entropy
loss_function = CrossEntropyLoss()
# set up the optimizer as Adam
optimizer = Adam(model.parameters(), lr=0.0001)

best_acc = 0.0

# do the training
for epoch in range(params_training["epochs_max"]):
    model.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = model(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss + loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    print()

# save the after training weight
time = datetime.now().strftime("%Y%m%d%H%M%S")  # e.g. 2023-01-01 00:00:00 -> resnet152-20230101000000.pth
save(path_weight, "resnet152-" + time + ".pth")
