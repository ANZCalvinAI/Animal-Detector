from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet152

# ====================
# customise parameters
# ====================
params = {
    "num_classes": 10,  # Number of classes
    "bs": 32,           # Training batch size
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
# Load pretrained ResNet50 Model
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
