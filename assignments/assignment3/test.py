import os
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics import accuracy_score
from skimage import feature
import cv2
import numpy as np
from dataloader import create_dataloaders

# Function to extract features using the provided feature extractor
def extract_features(feature_extractor, dataloader):
    feature_extractor.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs, lbls = inputs.to(device), lbls.to(device)
            # Feature extraction using the specified feature extractor
            extracted_features = feature_extractor(inputs).squeeze().cpu().numpy()
            features.append(extracted_features)
            labels.append(lbls.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# Function to compute LBP features
def compute_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize
    return hist

# Function to extract LBP features
def extract_lbp_features(dataloader):
    features = []
    labels = []

    for images, lbls in dataloader:
        for img in images:
            lbp_features = compute_lbp_features(img.permute(1, 2, 0).numpy())
            features.append(lbp_features)
        labels.append(lbls.numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# Function to evaluate recognition performance
def evaluate_performance(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DataLoaders
    base_path = os.path.join('datasets', 'ears', 'images-cropped')
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    test_dir = os.path.join(base_path, 'test')
    _, _, test_loader, num_classes = create_dataloaders(train_dir, val_dir, test_dir)

    # Load the pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    # Remove the classification layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    # Extract features
    resnet50_features, labels = extract_features(feature_extractor, test_loader)

    # Extract LBP features
    lbp_features, _ = extract_lbp_features(test_loader)

    # Combine features (you can use more sophisticated fusion methods)
    combined_features = np.concatenate([resnet50_features, lbp_features], axis=1)

    # Dummy recognition predictions (replace this with your recognition method)
    predictions = np.argmax(combined_features, axis=1)

    # Evaluate recognition performance
    evaluate_performance(predictions, labels)
