from os import walk
from typing import Tuple, Any


class FileManager(object):
    @staticmethod
    def find_cascade_filenames(base_path: str) -> [str]:
        """
        Reads the filenames from the given identities.txt file on base_path.
        :param base_path: path to the directory which holds identities.txt file.
        :return: a list of cascade file paths.
        """
        lines = []
        with open(base_path+'identites.txt', "r") as file:
            # Reading from a file
            lines = file.readlines()
        return [base_path+'ears/'+(line.split(" ")[0].split(".png")[0]) for line in lines]

    @staticmethod
    def read_ground_truths(filenames: [str]) -> {str}:
        """
        Reads the files containing grounds truths and saved them in a dictionary.
        :param filenames: a list of files contain grounds truths.
        :return: a dictionary of ground truths.
        """
        grounds_truths = {}
        for filename in filenames:
            full_filename = filename+'.txt'
            with open(full_filename, "r") as file:
                # Reading from a file
                line = file.readline()
                _, x, y, w, h = line.split()
                grounds_truths[filename] = (x, y, w, h)
        return grounds_truths


    @staticmethod
    def prepare_data(base_path: str) -> tuple[[str], [str], [str], {str}]:
        """
        Prepares the data for training and testing.
        :param base_path: path to the directory whihc holds identities.txt file.
        :return: train and test sets with cascade file paths.
        """
        filenames = FileManager.find_cascade_filenames(base_path=base_path)
        train_set, test_set = FileManager.split_data_set(filenames=filenames, train_ratio=0.1)
        ground_truths = FileManager.read_ground_truths(filenames=filenames)

        return filenames, train_set, test_set, ground_truths

    @staticmethod
    def split_data_set(filenames: [str], train_ratio: float = 0.8):
        """
        Splits the given cascade file paths into train and test sets.
        :param filenames: a list of cascade file paths.
        :param train_ratio: percentage of the train set.
        :return: a tuple of train and test sets.
        """
        train_size = int(len(filenames) * train_ratio)

        train_set = filenames[0:train_size]
        test_set = filenames[train_size:]

        return train_set, test_set

