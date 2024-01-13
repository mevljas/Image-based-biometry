import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils.evaluator import Evaluator


class Hog(object):

    @staticmethod
    def test(data_path: str, filenames: dict, hog_extractor) -> (int, [], []):
        """
        Test the resnet with HOG features.
        :param filenames: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        logging.debug(f'Testing Resnet HOG.')

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('.')[0], f) for (dirpath, dirnames, files) in os.walk(data_path) for
                 f in
                 files]

        image_features = []
        image_names = []
        for image_path, image_name, _ in files:
            logging.debug('Calculating HOG for: ' + image_path)

            # Open and preprocess the image
            image = np.array(Image.open(image_path).convert('RGB'))

            # Extract HOG features
            hog_features = hog_extractor.extract_features(image)

            image_features.append(hog_features)
            image_names.append(image_name)

        # Calculate the similarity matrix
        similarity_matrix = Evaluator.calculate_similarity_matrix(image_features)

        # Find the most similar image
        most_similar_image = Evaluator.find_most_similar_image(similarity_matrix)

        accuracy = Evaluator.calculate_accuracy(image_names=image_names,
                                                most_similar_image=most_similar_image,
                                                filenames=filenames)

        logging.debug('Finished testing Resnet with HOG features.')
        logging.debug('Resnet with HOG features accuracy: ' + str(accuracy))
        return accuracy, image_features, files
