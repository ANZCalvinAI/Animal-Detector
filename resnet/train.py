from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models

# ====================
# customise parameters
# ====================
params = {
    "num_classes": 10,  # Number of classes
    "bs": 32,           # Training batch size
    "epochs_max": 3     # Training maximal epochs
}

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
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}

# ============
# Data Loading
# ============
# Set train and valid directory paths
train_directory = 'train'
valid_directory = 'test'
# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
# Create iterators for the Data loaded using DataLoader module
train_data = DataLoader(data['train'], batch_size=params["bs"], shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=params["bs"], shuffle=True)
test_data = DataLoader(data['test'], batch_size=params["bs"], shuffle=True)
# Print the train, validation and test set data sizes
print(f"train data size\n{train_data_size}\n")
print(f"valid data size\n{valid_data_size}\n")
print(f"test data size\n{test_data_size}\n")

# =================
# Transfer Learning
# =================
# Load pretrained ResNet152 Model
resnet152 = models.resnet152(weights=None)

# Freeze model parameters
for param in resnet152.parameters():
    param.requires_grad = False

# Change the final layer of ResNet152 Model for Transfer Learning
fc_inputs = resnet152.fc.in_features
resnet152.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, params["num_classes"]),
    nn.LogSoftmax(dim=1)  # For using NLLLoss()
)

# Convert model to be used on GPU
resnet152 = resnet152.to('cuda:0')

# Define Optimizer and Loss Function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet152.parameters())

# ========
# Training
# ========
for epoch in range(params["epochs_max"]):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        # Set to training mode
        model.train()
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
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
            print(
                "Batch number: {:03d},
                Training: Loss: {:.4f},
                Accuracy: {:.4f}".format(i, loss.item(), acc.item())
            )
