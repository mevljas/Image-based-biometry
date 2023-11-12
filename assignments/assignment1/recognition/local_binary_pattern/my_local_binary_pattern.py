import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class MyLocalBinaryPattern(object):
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
                    pixel_value = MyLocalBinaryPattern.get_lbp_pixel(img, center, x, y)
                    binary_values.append(pixel_value)
                decimal_value = sum([binary_values[p] * (2 ** p) for p in range(num_points)])
                if uniform and MyLocalBinaryPattern.is_uniform(binary_values):
                    lbp_image[i, j] = decimal_value
                else:
                    lbp_image[i, j] = num_points + 1  # Mark non-uniform patterns

        return lbp_image

    @staticmethod
    def is_uniform(binary_values):
        transitions = sum((a != b) for a, b in zip(binary_values, binary_values[1:]))
        return transitions <= 2

    @staticmethod
    def draw_histograms(histograms: list):
        output_list_len = len(histograms)
        for i in range(output_list_len):
            (lbp_image, title, description) = histograms[i]
            plt.gca().set_position((.1, .3, .8, .6))  # to make a bit of room for extra text
            plt.plot(cv2.calcHist([lbp_image], [0], None, [256], [0, 256]), color="black")
            plt.xlim([0, 260])
            plt.title(title)
            plt.xlabel("Bins")
            plt.ylabel("Number of pixels")
            plt.figtext(.02, .02,
                        description)
            plt.show()

    @staticmethod
    def save_histograms(histograms: list):
        dir = 'output/local_binary_pattern/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        output_list_len = len(histograms)
        for i in range(output_list_len):
            (lbp_image, title, description, filename) = histograms[i]
            plt.gca().set_position((.1, .3, .8, .6))  # to make a bit of room for extra text
            plt.plot(cv2.calcHist([lbp_image], [0], None, [256], [0, 256]), color="black")
            plt.xlim([0, 260])
            plt.title(title)
            plt.xlabel("Bins")
            plt.ylabel("Number of pixels")
            plt.figtext(.02, .02,
                        description)
            plt.savefig(dir + filename)

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

        histograms = []

        for filename in filenames:
            logging.debug('Calculating LBP for: ' + filename)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            # radii = [1, 2, 3]
            # code_lengths = [8, 16]
            # overlaps = [1, 2]
            # uniform_options = [True, False]

            radii = [2]
            code_lengths = [8]
            overlaps = [1]
            uniform_options = [True]

            for radius in radii:
                for code_length in code_lengths:
                    for overlap in overlaps:
                        for uniform_option in uniform_options:
                            logging.debug('Calculating LBP for: ' + filename +
                                          ' with parameters: radius: ' + str(radius) +
                                          ', code_length: ' + str(code_length) +
                                          ', overlap: ' + str(overlap) +
                                          ', uniform_option: ' + str(uniform_option) + '.')
                            lbp_image = MyLocalBinaryPattern.calculate_lbp(img, radius, code_length, uniform_option)
                            title = f"LBP {filename.split('/')[-1]}"
                            description = f"""
                            
                            Radius: {radius}, Code Length: {code_length}, Overlap: {overlap}, Uniform: {uniform_option}"""
                            histograms.append((lbp_image, title, description, filename.split("/")[-1]))

        logging.debug('Finished calculating LBP for all images.')
        # LocalBinaryPattern.draw_histograms(histograms)
        MyLocalBinaryPattern.save_histograms(histograms=histograms)
