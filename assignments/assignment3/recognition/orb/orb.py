import logging
import os

import numpy as np
from PIL import Image

from utils.evaluator import Evaluator


class Orb(object):

    @staticmethod
    def test(data_path: str, filenames: dict, orb_extractor) -> (int, [], []):
        """
        Test the resnet with ORB features.
        :param filenames: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        logging.debug(f'Testing ORB.')

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('.')[0], f) for (dirpath, dirnames, files) in os.walk(data_path) for
                 f in
                 files]

        image_features = []
        image_names = []
        for image_path, image_name, _ in files:
            logging.debug('Calculating ORB for: ' + image_path)

            # Open and preprocess the image
            image = np.array(Image.open(image_path).convert('RGB'))

            # Extract ORB features
            orb_features = orb_extractor.extract_features(image)

            if orb_features is not None and len(orb_features) > 0:
                image_features.append(orb_features)
            image_names.append(image_name)

        # Calculate the similarity matrix
        similarity_matrix = Evaluator.calculate_similarity_matrix(image_features)

        # Find the most similar image
        most_similar_image = Evaluator.find_most_similar_image(similarity_matrix)

        accuracy = Evaluator.calculate_accuracy(image_names=image_names,
                                                most_similar_image=most_similar_image,
                                                filenames=filenames)

        logging.debug('Finished testing ORB.')
        logging.debug('ORB accuracy: ' + str(accuracy))
        return accuracy, image_features, files
