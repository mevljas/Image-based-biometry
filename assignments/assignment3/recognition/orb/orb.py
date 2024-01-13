import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils.evaluator import Evaluator


class Orb(object):

    @staticmethod
    def test(data_path: str, filenames: dict,
                        model, orb_extractor) -> (int, [], []):
        """
        Test the resnet with ORB features.
        :param filenames: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        logging.debug(f'Testing Resnet ORB.')

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('.')[0], f) for (dirpath, dirnames, files) in os.walk(data_path) for
                 f in
                 files]

        image_features = []
        image_names = []
        for image_path, image_name, _ in files:
            logging.debug('Calculating Resnet ORB for: ' + image_path)

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

            # Ensure image is a valid NumPy array
            if image is not None:
                # Extract ORB features
                orb_features = orb_extractor.extract_features(np.array(image))
                image_features.append(orb_features)

            image_names.append(image_name)

        # Calculate the similarity matrix
        similarity_matrix = Evaluator.calculate_similarity_matrix(image_features)

        # Find the most similar image
        most_similar_image = Evaluator.find_most_similar_image(similarity_matrix)

        accuracy = Evaluator.calculate_accuracy(image_names=image_names,
                                             most_similar_image=most_similar_image,
                                             filenames=filenames)

        logging.debug('Finished testing Resnet with ORB features.')
        logging.debug('Resnet with ORB features accuracy: ' + str(accuracy))
        return accuracy, image_features, files