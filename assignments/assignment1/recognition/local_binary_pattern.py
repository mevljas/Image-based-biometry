import logging
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from recognition.scikit_local_binary_pattern import ScikitLocalBinaryPattern


class LocalBinaryPattern(object):
    @staticmethod
    def draw_histogram(lbp_image, title):
        hist, _ = np.histogram(lbp_image, bins=np.arange(0, 256), range=(0, 256))
        plt.bar(np.arange(0, 256), hist, width=1.0, color='gray')
        plt.title(title)
        plt.show()

    def calculate_similarity_matrix(images):
        num_images = len(images)
        similarity_matrix = np.zeros((num_images, num_images))

        for i in range(num_images):
            for j in range(num_images):
                # Calculate cosine similarity between LBP histograms
                similarity_matrix[i, j] = cosine_similarity([images[i]], [images[j]])[0, 0]

        return similarity_matrix

    def find_most_similar_image(similarity_matrix, image_names):
        num_images = similarity_matrix.shape[0]
        most_similar_image = np.zeros(num_images, dtype=int)

        for i in range(num_images):
            # Exclude the image itself from the comparison
            sim_values = np.delete(similarity_matrix[i, :], i)
            most_similar_image[i] = np.argmax(sim_values)

        return most_similar_image

    @staticmethod
    def train_local_binary_pattern(data_path: str, identities: dict) -> (int, int, int):
        """
        Trains the local_binary_pattern model with different parameters and find parameters with the highest accuracy.
        :param identities: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('_')[0]) for (dirpath, dirnames, files) in os.walk(data_path) for
                 f in
                 files]

        best_accuracy = 0
        best_parameters = (0, 0, 0)

        # for radius in [1, 2, 3]:
        for radius in [1]:
            # for n_points in [8, 16]:
            for n_points in [16]:
                # for uniform_option in [True, False]:
                for uniform_option in [True]:
                    lbp_features = []
                    image_names = []
                    for image_path, image_name in files:
                        logging.debug('Calculating LBP for: ' + image_path)
                        # Read and resize images to a consistent size
                        img = cv2.resize(cv2.imread(image_path), (64, 64))

                        logging.debug('Calculating LBP for: ' + image_path +
                                      ' with parameters: radius: ' + str(radius) +
                                      ', n_points: ' + str(n_points) +
                                      ', uniform_option: ' + str(uniform_option) + '.')
                        lbp_features.append(
                            ScikitLocalBinaryPattern.local_binary_pattern(img, n_points, radius, uniform_option))
                        image_names.append(image_name)

                    # Calculate the similarity matrix
                    similarity_matrix = LocalBinaryPattern.calculate_similarity_matrix(lbp_features)

                    # Find the most similar image for each image
                    most_similar_image = LocalBinaryPattern.find_most_similar_image(similarity_matrix, image_names)

                    # Print the results
                    # for i, similar_image_index in enumerate(most_similar_image):
                    #     logging.debug(f"{image_names[i]} is most similar to {image_names[similar_image_index]}")

                    correct_recognitions = 0
                    all_recognitions = 0
                    for i, similar_image_index in enumerate(most_similar_image):
                        query = image_names[i]
                        match = image_names[similar_image_index]
                        logging.debug(f"{query} is most similar to {match}")
                        all_recognitions += 1
                        if identities[query] == identities[match]:
                            correct_recognitions += 1

                    accuracy = correct_recognitions / all_recognitions
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameters = (radius, n_points, uniform_option)
                        logging.info('New best accuracy: ' + str(best_accuracy) + ' with parameters: '
                                     + str(best_parameters))

        logging.debug('Finished training LBP.')
        logging.info('Best accuracy: ' + str(best_accuracy) + ' with parameters: ' + str(best_parameters))
        return best_parameters
