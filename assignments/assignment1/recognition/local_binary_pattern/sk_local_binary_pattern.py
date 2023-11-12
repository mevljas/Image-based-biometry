import os

import numpy as np
from skimage import feature
from skimage import io
from skimage.color import rgb2gray


class SkLocalBinaryPattern(object):
    @staticmethod
    def compute_lbp(image, radius, n_points):
        lbp = feature.local_binary_pattern(image, n_points, radius, method="uniform")
        return lbp.flatten()

    @staticmethod
    def extract_lbp_features(image_path, radii, n_points, overlap, word_lengths):
        image = io.imread(image_path)
        gray_image = rgb2gray(image)

        features = []
        for radius in radii:
            for n in n_points:
                for word_length in range(0, word_lengths + 1):
                    lbp = SkLocalBinaryPattern.compute_lbp(gray_image, radius, n)
                    hist, _ = np.histogram(lbp, bins=np.arange(0, word_length + 1), density=True)
                    features.extend(hist)

        return features

    @staticmethod
    def process_images(image_paths, radii, n_points, overlap, word_lengths):
        all_features = []
        for image_path in image_paths:
            features = SkLocalBinaryPattern.extract_lbp_features(image_path, radii, n_points, overlap, word_lengths)
            all_features.append(features)

        return np.array(all_features)

    @staticmethod
    def train_local_binary_pattern(data_path: str, ground_truths: {str}) -> (
            float, (float, int, int, int),
            dict[str, list[tuple[int, int, int, int, int, int]], {str}]
    ):
        """
        Trains the local_binary_pattern model with different parameters and find parameters with the highest average iou.
        :param data_path: base path for cascade files.
        :param ground_truths: ground truths for all images.
        :return:
        """

        # Find all files in the given directory
        filenames = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(data_path) for f in
                     filenames]

        # Example usage
        radii = [1, 2]
        n_points = [8, 16]
        overlap = 0.5  # Adjust overlap as needed
        word_lengths = 256  # Adjust word length as needed

        features = SkLocalBinaryPattern.process_images(filenames, radii, n_points, overlap, word_lengths)

        # Display the extracted features
        print("Extracted features:")
        print(features)

        # You can now use these features for further analysis or machine learning tasks.
