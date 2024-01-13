import logging
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from skimage import feature
import numpy as np
import os


class Resnet(object):

    @staticmethod
    def calculate_similarity_matrix(images: list):
        num_images = len(images)
        similarity_matrix = np.zeros((num_images, num_images))

        for i in range(num_images):
            for j in range(num_images):
                # Calculate cosine similarity between LBP histograms
                similarity_matrix[i, j] = cosine_similarity([images[i]], [images[j]])[0, 0]

        return similarity_matrix

    @staticmethod
    def find_most_similar_image(similarity_matrix):
        num_images = similarity_matrix.shape[0]
        most_similar_image = np.zeros(num_images, dtype=int)

        for i in range(num_images):
            # Exclude the image itself from the comparison
            sim_values = np.delete(similarity_matrix[i, :], i)
            most_similar_image[i] = np.argmax(sim_values)

        return most_similar_image

    @staticmethod
    def calculate_accuracy(image_names: list, most_similar_image, filenames: dict):
        correct_recognitions = 0
        all_recognitions = 0
        for i, similar_image_index in enumerate(most_similar_image):
            query = image_names[i]
            match = image_names[similar_image_index]
            logging.debug(f"{query} is most similar to {match}")
            all_recognitions += 1
            if filenames[query] == filenames[match]:
                correct_recognitions += 1

        if correct_recognitions == 0:
            accuracy = 0
        else:
            accuracy = correct_recognitions / all_recognitions
        return accuracy

    @staticmethod
    def test_resnet(data_path: str, filenames: dict,
                    model) -> (int, [], []):
        """
        Test the resnet.
        :param filenames: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        logging.debug(f'Testing Resnet .')

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('.')[0], f) for (dirpath, dirnames, files) in os.walk(data_path) for
                 f in
                 files]

        image_features = []
        image_names = []
        for image_path, image_name, _ in files:
            logging.debug('Calculating Resnet for: ' + image_path)

            # Open and preprocess the image
            image = Image.open(image_path).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = preprocess(image)
            image = image.unsqueeze(0)

            # Extract features from the layer preceding the final softmax layer
            with torch.no_grad():
                features = model(image)
                image_features.append(np.array(features.squeeze()))

            image_names.append(image_name)

        # Calculate the similarity matrix
        similarity_matrix = Resnet.calculate_similarity_matrix(image_features)

        # Find the most similar image
        most_similar_image = Resnet.find_most_similar_image(similarity_matrix)

        accuracy = Resnet.calculate_accuracy(image_names=image_names,
                                             most_similar_image=most_similar_image,
                                             filenames=filenames)

        logging.debug('Finished testing LBP.')
        logging.debug('LBP accuracy: ' + str(accuracy) + ' %.')
        return accuracy, image_features, files
