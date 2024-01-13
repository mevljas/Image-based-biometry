import logging
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recognition.lbp.scikit_local_binary_pattern import ScikitLocalBinaryPattern


class Evaluator(object):

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

