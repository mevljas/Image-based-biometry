import os
import torch
import torchvision.models as models
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from PIL import Image
from skimage import feature
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pretrained ResNet50 model
model = models.resnet50(pretrained=False)

# Number of classes you want (e.g., 136)
num_classes = 136

# Modify the last fully connected layer to match the desired number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the state dictionary from the saved model
checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))

# Load the state dictionary into the modified model
model.load_state_dict(checkpoint, strict=False)

# Move the model to the device (CPU or GPU)
model = model.to(device)

model.eval()

# Feature extraction function
def extract_resnet_features(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Extract features from the layer preceding the final softmax layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        features = feature_extractor(image)

    return features.squeeze().cpu().numpy()

# LBP feature extraction function
def extract_lbp_features(image_path):
    image = read_image(image_path).float()
    image = image.mean(dim=0)  # Convert to grayscale
    lbp = feature.local_binary_pattern(image.numpy(), P=8, R=1, method="uniform")
    histogram, _ = np.histogram(lbp, bins=np.arange(0, 10), density=True)
    return histogram


# Set paths for the test set
test_dir = os.path.join('datasets', 'ears', 'images-cropped', 'test')

# Extract and store ResNet features
resnet_features = []
for filename in os.listdir(test_dir):
    image_path = os.path.join(test_dir, filename)
    features = extract_resnet_features(image_path)
    resnet_features.append(features)

# Store ResNet features (resnet_features) as needed for further recognition tasks

# Extract and store LBP features
lbp_features = []
for filename in os.listdir(test_dir):
    image_path = os.path.join(test_dir, filename)
    features = extract_lbp_features(image_path)
    lbp_features.append(features)

# Store LBP features (lbp_features) as needed for further recognition tasks
