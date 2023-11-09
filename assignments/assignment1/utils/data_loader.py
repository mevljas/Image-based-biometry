import logging


class FileManager(object):
    @staticmethod
    def find_ear_filenames(base_path: str) -> [str]:
        """
        Reads the filenames from the given identities.txt file on base_path.
        :param base_path: path to the directory which holds identities.txt file.
        :return: a list of cascade file paths.
        """
        logging.debug('Reading ear filenames from: ' + base_path)
        lines = []
        with open(base_path + 'identites.txt', "r") as file:
            # Reading from a file
            lines = file.readlines()
        logging.debug('Found ' + str(len(lines)) + ' ear filenames.')
        return [base_path + 'ears/' + (line.split(" ")[0].split(".png")[0]) for line in lines]

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
            full_filename = filename + '.txt'
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
    def prepare_data(base_path: str) -> tuple[[str], [str], [str], {str}]:
        """
        Prepares the data for training and testing.
        :param base_path: path to the directory whihc holds identities.txt file.
        :return: train and test sets with cascade file paths.
        """
        logging.debug('Preparing data from: ' + base_path)
        filenames = FileManager.find_ear_filenames(base_path=base_path)
        train_set, test_set = FileManager.split_data_set(filenames=filenames, train_ratio=0.1)
        ground_truths = FileManager.load_ground_truths(filenames=train_set)
        logging.debug('Prepared data.')

        return filenames, train_set, test_set, ground_truths

    @staticmethod
    def split_data_set(filenames: [str], train_ratio: float = 0.8):
        """
        Splits the given cascade file paths into train and test sets.
        :param filenames: a list of cascade file paths.
        :param train_ratio: percentage of the train set.
        :return: a tuple of train and test sets.
        """
        logging.debug('Splitting data set into train and test sets with ratio: ' + str(train_ratio) + '.')
        train_size = int(len(filenames) * train_ratio)

        train_set = filenames[0:train_size]
        test_set = filenames[train_size:]

        logging.debug('Set sizes: train: ' + str(len(train_set)) + ', test: ' + str(len(test_set)) + '.')

        return train_set, test_set
