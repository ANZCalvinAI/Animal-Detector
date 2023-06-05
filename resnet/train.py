from json import dumps
from torch import device, cuda, load, save
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import resnet152
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize,\
    CenterCrop
from torchvision.datasets import ImageFolder
from datetime import datetime

# ====================
# customise parameters
# ====================
# project path parameter
project_path = "C:/Users/cz199/PycharmProjects/Animal-Detector/"

# =====================
# load and process data
# =====================
data_transform = {
    "train": Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    "val": Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}

image_path = project_path + "datasets/iNat2021/images/"

train_dataset = ImageFolder(
    root=image_path + "train",
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

batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_dataset = ImageFolder(
    root=image_path + "val",
    transform=data_transform["val"]
)
val_num = len(val_dataset)
validate_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# =======================
# set up device and model
# =======================
# set up the device as GPU as first choice and CPU as alternative
device = device("cuda:0" if cuda.is_available() else "cpu")
print(f"device\n{device}\n")

# set up the model as ResNet 152
model = resnet152(weights=None)
print(f"ResNet model architecture\n{model}\n")

# set up the model weight as the default weight, i.e. "resnet152-19000101000000.pth"
model.to(device)
model_weight_path = project_path + "resnet/weights/resnet152-19000101000000.pth"
missing_keys, unexpected_keys = model.load_state_dict(
    load(model_weight_path),
    strict=False
)
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
for epoch in range(3):
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
time = datetime.now().strftime("%Y%m%d%H%M%S")
save(model.state_dict(), "weights/resnet152-" + time + ".pth")
