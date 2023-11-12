import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class LocalBinaryPattern(object):
    @staticmethod
    def get_lbp_pixel(img, center, x, y):
        pixel_value = 0
        if img[x, y] >= center:
            pixel_value = 1
        return pixel_value

    @staticmethod
    def calculate_lbp(img, radius, num_points, uniform=True):
        lbp_image = np.zeros_like(img, dtype=np.uint8)

        for i in range(radius, img.shape[0] - radius):
            for j in range(radius, img.shape[1] - radius):
                center = img[i, j]
                binary_values = []
                for k in range(num_points):
                    x = int(i + radius * np.cos(2 * np.pi * k / num_points))
                    y = int(j - radius * np.sin(2 * np.pi * k / num_points))
                    pixel_value = LocalBinaryPattern.get_lbp_pixel(img, center, x, y)
                    binary_values.append(pixel_value)
                decimal_value = sum([binary_values[p] * (2 ** p) for p in range(num_points)])
                if uniform and LocalBinaryPattern.is_uniform(binary_values):
                    lbp_image[i, j] = decimal_value
                else:
                    lbp_image[i, j] = num_points + 1  # Mark non-uniform patterns

        return lbp_image

    @staticmethod
    def is_uniform(binary_values):
        transitions = sum((a != b) for a, b in zip(binary_values, binary_values[1:]))
        return transitions <= 2

    @staticmethod
    def draw_histogram(lbp_image, title):
        hist, bins = np.histogram(lbp_image.flatten(), bins=range(0, 60), density=True)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.title(title)
        plt.show()

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

        for filename in filenames:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            radii = [1, 2, 3]
            code_lengths = [8, 16]
            overlaps = [1, 2]
            uniform_options = [True, False]

            for radius in radii:
                for code_length in code_lengths:
                    for overlap in overlaps:
                        for uniform_option in uniform_options:
                            lbp_image = LocalBinaryPattern.calculate_lbp(img, radius, code_length, uniform_option)
                            title = f"LBP - Radius: {radius}, Code Length: {code_length}, Overlap: {overlap}, Uniform: {uniform_option}"
                            LocalBinaryPattern.draw_histogram(lbp_image, title)
