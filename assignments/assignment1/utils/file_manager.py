from os import walk
from typing import Tuple, Any


class FileManager(object):
    @staticmethod
    def find_cascade_files(base_path: str) -> [str]:
        """
        Reads the filenames from the given identities.txt file on base_path.
        :param base_path: path to the directory whihc holds identities.txt file.
        :return: a list of cascade file paths.
        """
        lines = []
        with open(base_path+'identites.txt', "r") as file:
            # Reading from a file
            lines = file.readlines()
        return [base_path+line.split(" ")[0] for line in lines]

    @staticmethod
    def prepare_data(base_path: str) -> tuple[[str], [str]]:
        """
        Prepares the data for training and testing.
        :param base_path: path to the directory whihc holds identities.txt file.
        :return: train and test sets with cascade file paths.
        """
        cascade_filenames = FileManager.find_cascade_files(base_path=base_path)
        train_set, test_set = FileManager.split_data_set(cascade_filenames=cascade_filenames, train_ratio=0.8)

        return train_set, test_set

    @staticmethod
    def split_data_set(cascade_filenames: [str], train_ratio: float = 0.8):
        """
        Splits the given cascade file paths into train and test sets.
        :param cascade_filenames: a list of cascade file paths.
        :param train_ratio: percentage of the train set.
        :return: a tuple of train and test sets.
        """
        train_size = int(len(cascade_filenames) * train_ratio)

        train_set = cascade_filenames[0:train_size]
        test_set = cascade_filenames[train_size:]

        return train_set, test_set

