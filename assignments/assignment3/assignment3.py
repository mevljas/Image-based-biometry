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
from sklearn.metrics.pairwise import cosine_similarity

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


def find_most_similar_image(similarity_matrix):
    num_images = similarity_matrix.shape[0]
    most_similar_image = np.zeros(num_images, dtype=int)

    for i in range(num_images):
        # Exclude the image itself from the comparison
        sim_values = np.delete(similarity_matrix[i, :], i)
        most_similar_image[i] = np.argmax(sim_values)

    return most_similar_image

def calculate_accuracy(image_names: list, most_similar_image, filenames: dict):
    correct_recognitions = 0
    all_recognitions = 0
    for i, similar_image_index in enumerate(most_similar_image):
        query = image_names[i]
        match = image_names[similar_image_index]
        print(f"{query} is most similar to {match}")
        all_recognitions += 1
        if filenames[query] == filenames[match]:
            correct_recognitions += 1

    if correct_recognitions == 0:
        accuracy = 0
    else:
        accuracy = correct_recognitions / all_recognitions
    return accuracy


def calculate_similarity_matrix(images: list):
    num_images = len(images)
    similarity_matrix = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(num_images):
            # Calculate cosine similarity between LBP histograms
            similarity_matrix[i, j] = cosine_similarity([images[i]], [images[j]])[0, 0]

    return similarity_matrix



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

# Calculate the similarity matrix
similarity_matrix = calculate_similarity_matrix(lbp_features)

# Find the most similar image
most_similar_image = find_most_similar_image(similarity_matrix)

accuracy = calculate_accuracy(image_names=image_names,
                                                 most_similar_image=most_similar_image,
                                                 filenames=filenames)

logging.debug('Finished testing LBP.')
logging.debug('LBP accuracy: ' + str(accuracy))
