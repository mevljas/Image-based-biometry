import logging
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class PixelToPixel(object):
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
    def test(data_path: str, identities: dict) -> (int, int, int):
        """
        Test the pixel to pixel model and returns accuracy.
        :param identities: dictionary of filenames and their identities.
        :param data_path: base path for cascade files.
        :return:
        """

        # Find all files in the given directory
        files = [(os.path.join(dirpath, f), f.split('_')[0]) for (dirpath, dirnames, files) in os.walk(data_path) for
                 f in
                 files]

        images_vectors = []
        image_names = []

        for image_path, image_name in files:
            logging.debug('Calculating P2P for: ' + image_path)
            # Load and flatten images
            img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (64, 64)).flatten()
            images_vectors.append(img)
            image_names.append(image_name)

        # Convert image vectors to a 2D array
        image_matrix = np.vstack(images_vectors)

        # Calculate pairwise cosine similarity
        similarity_matrix = cosine_similarity(image_matrix)

        # Find the most similar image for each image
        most_similar_image = PixelToPixel.find_most_similar_image(similarity_matrix, image_names)

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

        logging.debug('Finished calculating P2P.')
        logging.info('P2P accuracy: ' + str(accuracy))

        return accuracy
