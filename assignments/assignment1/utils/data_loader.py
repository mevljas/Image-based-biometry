import logging
import os

import cv2


class FileManager(object):
    @staticmethod
    def load_identities(base_path: str) -> (dict[[str]], dict[str]):
        """
        Reads the filenames and identities from the given identities.txt file on base_path.
        :param base_path: path to the directory which holds identities.txt file.
        :return: a lists of filenames and identities.
        """
        logging.debug('Reading ear filenames from: ' + base_path)
        lines = []
        with open(base_path + 'identities.txt', "r") as file:
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
    def load_ground_truths(filenames: [str]) -> {str}:
        """
        Reads the files containing grounds truths and saves them in a dictionary.
        :param filenames: a list of files that contain grounds truths.
        :return: a dictionary of read ground truths.
        """
        logging.debug('Loading ground truths from.')
        grounds_truths = dict()
        for filename in filenames:
            full_filename = 'data/ears/' + filename + '.txt'
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
    def prepare_data(data_path: str, train_ratio: float = 0.8) -> tuple[[str], [str], [str], {list}, dict, dict]:
        """
        Prepares the data for training and testing.
        :param data_path: path to the directory which holds identities.txt file.
        :param train_ratio: percentage of the train set.
        :return: train and test sets with cascade file paths.
        """
        logging.debug('Preparing data from: ' + data_path)
        filenames, identities = FileManager.load_identities(base_path=data_path)
        train_set, test_set = FileManager.split_data_set(identities=identities, train_ratio=train_ratio)
        ground_truths = FileManager.load_ground_truths(filenames=filenames.keys())
        train_ground_truths, test_ground_truths = FileManager.split_ground_truths(ground_truths=ground_truths,
                                                                                  train_set=train_set)
        logging.debug('Prepared data.')

        return filenames, train_set, test_set, identities, train_ground_truths, test_ground_truths

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
        Splits the given cascade file paths into train and test sets.
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
