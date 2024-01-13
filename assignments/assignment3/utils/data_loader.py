import logging
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


class FileManager(object):
    @staticmethod
    def load_identities() -> (dict[[str]], dict[str]):
        """
        Reads the filenames and identities from the identities_test.txt file.
        :return: lists of filenames and identities.
        """
        logging.debug('Reading identities.')
        lines = []
        with open('identities_test.txt', "r") as file:
            # Reading from a file
            lines = file.readlines()
        logging.debug('Found ' + str(len(lines)) + ' ear filenames.')

        identities = dict()
        filenames = dict()
        for line in lines:
            filename, identity = line.strip().split(" ")
            short_filename = filename.split(".png")[0]

            filenames[short_filename] = identity
            if identity in identities:
                identities[identity].append(short_filename)
            else:
                identities[identity] = [short_filename]

        return filenames, identities

    @staticmethod
    def load_ground_truths(filenames: [str], labels_path: str) -> {str}:
        """
        Reads the files containing grounds truths and saves them in a dictionary.
        :param labels_path: a path to the directory which holds labels.
        :param filenames: a list of files that contain grounds truths.
        :return: a dictionary of read ground truths.
        """
        logging.debug('Loading ground truths.')
        grounds_truths = dict()
        for filename in filenames:
            full_filename = labels_path + filename + '.txt'
            with (open(full_filename, "r") as file):
                line = file.readline()
                _, x, y, width, height = line.split()
                square = grounds_truths.get(filename, [])
                square.append((float(x), float(y), float(width), float(height)))
                grounds_truths[filename] = square

            logging.debug('Loaded ' + str(len(grounds_truths.get(filename, []))) +
                          ' squares for gt file: ' + filename + '.')

        logging.debug('Loaded ' + str(len(grounds_truths.keys())) + ' ground truth files.')
        return grounds_truths

    @staticmethod
    def prepare_data(labels_path: str) -> tuple[dict[[str]], dict[str], {str}]:
        """
        Prepares the data for training and testing.
        :param data_path: path to the directory which holds images.
        :return: lists of images and identities.
        """
        logging.debug('Preparing data.')
        filenames, identities = FileManager.load_identities()
        ground_truths = FileManager.load_ground_truths(filenames=filenames.keys(), labels_path=labels_path)
        logging.debug('Prepared data.')

        return filenames, identities, ground_truths,

    @staticmethod
    def split_ground_truths(ground_truths: dict[str, list[tuple[int, int, int, int]]], train_set: [str]) -> \
            (dict[str, list[[int, int, int, int]]], dict[str, [(int, int, int, int)]]):
        """
        Splits the given ground truths into train and test sets.
        :param train_set: Set of train set filenames.
        :param ground_truths: a dictionary of ground truths.
        :return: a tuple of train and test sets.
        """
        logging.debug('Splitting ground truths into train and test sets.')
        train_ground_truths = dict()
        test_ground_truths = dict()
        for filename in ground_truths.keys():
            if filename in train_set:
                train_ground_truths[filename] = ground_truths[filename]
            else:
                test_ground_truths[filename] = ground_truths[filename]

        return train_ground_truths, test_ground_truths

    @staticmethod
    def split_data_set(identities: dict[str], train_ratio: float):
        """
        Splits the files of each identity into train and test sets.
        :param identities: a dictionary of identifies and their filenames.
        :param train_ratio: percentage of the train set.
        :return: a tuple of train and test sets.
        """
        logging.debug('Splitting data set into train and test sets with ratio: ' + str(train_ratio) + '.')

        train_set = []
        test_set = []

        for identity, filenames in identities.items():
            train_size = int(len(filenames) * train_ratio)

            train_set.extend(filenames[0:train_size])
            test_set.extend(filenames[train_size:])

        logging.debug(
            'Set sizes: train: ' + str(len(train_set)) + ', test: ' + str(len(test_set)) + '.')

        return train_set, test_set

    @staticmethod
    def crop_image(image, image_name: str, save_directory: str, x: float, y: float, x2: float, y2: float):
        """
        Crops the given image with the given parameters and saves it.
        :param image_name: name of the image to be cropped.
        :param save_directory: directory to save the cropped image.
        :param image: image to be cropped.
        :param x: x coordinate of the top left corner of the crop.
        :param y: y coordinate of the top left corner of the crop.
        :param x2: x2 coordinate of the bottom right corner of the crop.
        :param y2: y2 coordinate of the bottom right corner of the crop.
        """
        logging.debug('Cropping image: ' + image_name + ' to position: ' + str(x) + ', ' + str(y) +
                      ', ' + str(x2) + ', ' + str(y2) + '.')
        cropped_image = image[int(y):int(y2), int(x):int(x2)]
        cv2.imwrite(filename=save_directory + image_name, img=cropped_image)

    @staticmethod
    def save_full_image(image, image_name: str, save_directory: str):
        """
        Saves the whole given image.
        :param image_name: name of the image to be cropped.
        :param save_directory: directory to save the cropped image.
        :param image: image to be saved.
        """
        logging.debug('Saving whole image: ' + image_name + '.')
        cv2.imwrite(filename=save_directory + image_name, img=image)

    @staticmethod
    def save_images(detections: dict[str, list[tuple[int, int, int, int]]],
                    grounds_truths: {str},
                    save_directory: str):
        """
        Saves detected and ground truths image regions.
        :param detections: dictionary of detected regions per image.
        :param save_directory: directory to save the cropped images.
        :param grounds_truths: dictionary of ground truths for the images.
        :return:
        """
        logging.debug('Saving images.')
        detected_path = save_directory + 'detected/'
        ground_truths_path = save_directory + 'ground_truths/'

        if not os.path.exists(detected_path):
            os.makedirs(detected_path)

        if not os.path.exists(ground_truths_path):
            os.makedirs(ground_truths_path)

        for image_name, ground_truth_squares in grounds_truths.items():
            image = cv2.imread('data/ears/' + image_name + '.png')
            square_counter = 0
            for ground_truth_square in ground_truth_squares:
                x, y, x2, y2 = ground_truth_square
                FileManager.crop_image(image=image,
                                       image_name=image_name + '_' + str(square_counter) + '.png',
                                       save_directory=ground_truths_path,
                                       x=x, y=y,
                                       x2=x2, y2=y2)
                square_counter += 1

            detection_squares = detections[image_name]
            square_counter = 0
            if detection_squares is not None:
                for detection_square in detection_squares:
                    x, y, x2, y2 = detection_square
                    FileManager.crop_image(image=image,
                                           image_name=image_name + '_' + str(square_counter) + '.png',
                                           save_directory=detected_path,
                                           x=x, y=y,
                                           x2=x2, y2=y2)
                    square_counter += 1
            else:
                FileManager.save_full_image(image=image,
                                            image_name=image_name + '_' + str(square_counter) + '.png',
                                            save_directory=detected_path)

        logging.debug('Saved images.')

    @staticmethod
    def save_lbp_histograms(directory_path: str, histograms, files: str):
        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        for file, lbp_image in zip(files, histograms):
            _, name, filename = file

            histogram, _ = np.histogram(lbp_image, bins=256, range=(0, 256))

            plt.bar(range(len(histogram)), histogram)
            plt.title(f'LBP Histogram - Image {name}')
            plt.xlabel('Bin')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(directory_path, f'{filename}.png'))
            plt.close()
