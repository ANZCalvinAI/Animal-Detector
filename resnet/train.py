import os
import torch
from torch import device, cuda, nn, optim, save
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from utils import get_weight_latest
import time
from datetime import datetime

# ====================
# customise parameters
# ====================
# project path parameter
path_project = "C:/Users/cz199/PycharmProjects/Animal-Detector/"

# training parameters
params = {
    "batch_size": 32,  # Training batch size
    "epochs_max": 3    # Training maximal epochs
}

# ===============
# path parameters
# ===============
path_weight = path_project + "resnet/weights/"
path_dataset = path_project + "datasets/iNat2021/images/"
path_train = path_dataset + "train/"
path_valid = path_dataset + "val/"
path_test = path_dataset + "test/"

# =================
# Data Augmentation
# =================
# Applying Transforms to the Data
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(
            size=256,
            scale=(0.8, 1.0)
        ),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

# ============
# Data Loading
# ============
# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=path_train, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=path_valid, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=path_test, transform=image_transforms['test'])
}

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

# Number of Classes
num_classes = len(data['train'].class_to_idx)

# Create iterators for the Data loaded using DataLoader module
batch_size = params["batch_size"]

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

# Print the train, validation and test set data sizes
print(f"train data size\n{train_data_size}\n")
print(f"valid data size\n{valid_data_size}\n")
print(f"test data size\n{test_data_size}\n")

# =============
# set up device
# =============
# set up the device as GPU as first choice and CPU as alternative
device = device("cuda:0" if cuda.is_available() else "cpu")
print(f"device\n{device}\n")

# =================
# Transfer Learning
# =================
"""
Original:
# Load pretrained ResNet152 Model
resnet152 = models.resnet152(pretrained=True)
"""
# if the weight folder has not been created
if not os.path.exists(path_weight):
    # create a weight folder, necessary for saving weights after training
    os.mkdir(path_weight)
    # load the model and the pretrained weight from Torch Hub
    resnet152 = models.resnet152(weights=None)
# if there is a weight folder
else:
    # if there is no weight file in the weight folder
    if not len(os.listdir(path_weight)):
        # load the model and the pretrained weight from Torch Hub
        resnet152 = models.resnet152(weights=None)
    # if there is one or more weight files in the weight folder
    else:
        # load the ResNet 152 model from local
        resnet152 = models.resnet152(weights=None)
        # search for the latest weight from local
        weight_latest = get_weight_latest(path_weight)
        # let state_dict load the latest weight from local
        state_dict = torch.load(path_weight + weight_latest)
        # let the ResNet 152 model load the state_dict
        resnet152.load_state_dict(state_dict)

# Freeze model parameters
for param in resnet152.parameters():
    param.requires_grad = False

# Change the final layer of ResNet152 Model for Transfer Learning
fc_inputs = resnet152.fc.in_features
resnet152.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1)  # For using NLLLoss()
)

"""
Original:
# Convert model to be used on GPU
resnet152 = resnet152.to('cuda:0')
"""
resnet152 = resnet152.to(device)

# Define Optimizer and Loss Function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet152.parameters())

# ========
# Training
# ========
epochs_max = params["epochs_max"]

for epoch in range(epochs_max):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch + 1, epochs_max))
    # Set to training mode
    resnet152.train()
    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0

    loss_criterion = nn.CrossEntropyLoss()

    for i, (inputs, labels) in enumerate(train_data):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Clean existing gradients
        optimizer.zero_grad()
        # Forward pass - compute outputs on input data using the model
        outputs = resnet152(inputs)
        # Compute loss
        loss = loss_criterion(outputs, labels)
        # Backpropagate the gradients
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)
        # Compute the accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)
        print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

# ==========
# Validation
# ==========
# Validation - No gradient tracking needed
history = []

with torch.no_grad():
    # Set to evaluation mode
    resnet152.eval()
    # Validation loop
    for j, (inputs, labels) in enumerate(valid_data):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass - compute outputs on input data using the model
        outputs = resnet152(inputs)
        # Compute loss
        loss = loss_criterion(outputs, labels)
        # Compute the total loss for the batch and add it to valid_loss
        valid_loss += loss.item() * inputs.size(0)
        # Calculate validation accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        # Compute total accuracy in the whole batch and add to valid_acc
        valid_acc += acc.item() * inputs.size(0)
        printed = "Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}"
        print(printed.format(j, loss.item(), acc.item()))

# Find average training loss and training accuracy
avg_train_loss = train_loss / train_data_size
avg_train_acc = train_acc / float(train_data_size)

# Find average validation loss and validation accuracy
avg_valid_loss = valid_loss/valid_data_size
avg_valid_acc = valid_acc / float(valid_data_size)

history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
epoch_end = time.time()

printed = "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, nttValidation: Loss: {:.4f}, Accuracy: {:.4f}%," \
          "Time: {:.4f}s"
print(
    printed.format(
        epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100, epoch_end - epoch_start
    )
)

# save the after training weight
time = datetime.now().strftime("%Y%m%d%H%M%S")  # e.g. 2023-01-01 00:00:00 -> resnet152-20230101000000.pth
save(resnet152.state_dict(), path_weight + "resnet152-" + time + ".pth")
