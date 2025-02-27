import logging
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recognition.custom_local_binary_pattern import CustomLocalBinaryPattern
from recognition.scikit_local_binary_pattern import ScikitLocalBinaryPattern


class LocalBinaryPattern(object):

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
    def train_local_binary_pattern(data_path: str, filenames: dict, use_scikit: bool) -> (int, int, int):
        """
        Trains the local binary pattern model with different parameters and
        find parameters with the highest accuracy.
        :param use_scikit: whether to use scikit or custom implementation.
        :param filenames: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        logging.debug(f'Training LBP with scikit: {use_scikit}.')

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('_')[0]) for (dirpath, dirnames, files) in os.walk(data_path) for
                 f in
                 files]

        best_accuracy = 0
        best_parameters = (0, 0, 0)

        # accuracy: 0.06793802145411204 with parameters: (3, 24, False)
        for radius in [1, 2, 3]:
            # for radius in [1]:
            for neighbor_points in range(8, 25, 8):
                # for neighbor_points in [8]:
                for uniform in [True, False]:
                    # for uniform in [True]:
                    image_features = []
                    image_names = []
                    for image_path, image_name in files:
                        logging.debug('Calculating LBP for: ' + image_path +
                                      ' with parameters: radius: ' + str(radius) +
                                      ', neighbor points: ' + str(neighbor_points) +
                                      ', uniform: ' + str(uniform) + '.')

                        # Read and resize images to a consistent size
                        img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (128, 128))

                        if use_scikit:
                            image_features.append(
                                ScikitLocalBinaryPattern.run(img, neighbor_points, radius, uniform))
                        else:
                            image_features.append(
                                CustomLocalBinaryPattern.run(img, radius, neighbor_points, uniform))

                        image_names.append(image_name)

                    # Calculate the similarity matrix
                    similarity_matrix = LocalBinaryPattern.calculate_similarity_matrix(image_features)

                    # Find the most similar image
                    most_similar_image = LocalBinaryPattern.find_most_similar_image(similarity_matrix)

                    accuracy = LocalBinaryPattern.calculate_accuracy(image_names=image_names,
                                                                     most_similar_image=most_similar_image,
                                                                     filenames=filenames)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameters = (radius, neighbor_points, uniform)
                        logging.info('New best accuracy: ' + str(best_accuracy) +
                                     ' with parameters: ' + str(best_parameters))

        logging.debug('Finished training LBP.')
        logging.debug('Best LBP accuracy: ' + str(best_accuracy) + ' with parameters: ' + str(best_parameters))
        return best_accuracy, best_parameters

    @staticmethod
    def test_local_binary_pattern(data_path: str, filenames: dict, use_scikit: bool,
                                  radius: int, neighbor_points: int, uniform: bool) -> (int, int, int):
        """
        Test the local binary pattern model with the provided parameters.
        :param use_scikit: whether to use scikit or custom implementation.
        :param filenames: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        logging.debug(f'Training LBP with scikit: {use_scikit}.')

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('_')[0], f) for (dirpath, dirnames, files) in os.walk(data_path) for
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

            if use_scikit:
                image_features.append(
                    ScikitLocalBinaryPattern.run(img, neighbor_points, radius, uniform))
            else:
                image_features.append(
                    CustomLocalBinaryPattern.run(img, radius, neighbor_points, uniform))

            image_names.append(image_name)

        # Calculate the similarity matrix
        similarity_matrix = LocalBinaryPattern.calculate_similarity_matrix(image_features)

        # Find the most similar image
        most_similar_image = LocalBinaryPattern.find_most_similar_image(similarity_matrix)

        accuracy = LocalBinaryPattern.calculate_accuracy(image_names=image_names,
                                                         most_similar_image=most_similar_image,
                                                         filenames=filenames)

        logging.debug('Finished testing LBP.')
        logging.debug('LBP accuracy: ' + str(accuracy))
        return accuracy, image_features, files
