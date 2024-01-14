import logging
import os

import cv2

from utils.evaluator import Evaluator

from skimage import feature


class LocalBinaryPattern(object):

    @staticmethod
    def test(data_path: str, filenames: dict,
             radius: int, neighbor_points: int, uniform: bool) -> (int, [], []):
        """
        Test the local binary pattern model with the provided parameters.
        :param filenames: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        logging.debug(f'Testing LBP with scikit.')

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('.')[0], f) for (dirpath, dirnames, files) in os.walk(data_path) for
                 f in
                 files]

        image_features = []
        image_names = []
        for image_path, image_name, _ in files:
            logging.debug('Calculating LBP for: ' + image_path +
                          ' with parameters: radius: ' + str(radius) +
                          ', neighbor points: ' + str(neighbor_points) +
                          ', uniform: ' + str(uniform) + '.')

            # Read and resize images to a consistent size
            img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (128, 128))

            features = feature.local_binary_pattern(img, P=neighbor_points, R=radius,
                                                    method='uniform' if uniform else 'default')
            # Correctly handle 3D and 2D features
            features = features.flatten() if features.ndim == 2 else features

            image_features.append(features)

            image_names.append(image_name)

        # Calculate the similarity matrix
        similarity_matrix = Evaluator.calculate_similarity_matrix(image_features)

        # Find the most similar image
        most_similar_image = Evaluator.find_most_similar_image(similarity_matrix)

        accuracy = Evaluator.calculate_accuracy(image_names=image_names,
                                                most_similar_image=most_similar_image,
                                                filenames=filenames)

        logging.debug('Finished testing LBP.')
        logging.debug('LBP accuracy: ' + str(accuracy))
        return accuracy, image_features, files
