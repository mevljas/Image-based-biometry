# This is a sample Python script.
import logging

import coloredlogs

from detection.viola_jones import detect_ears, calculate_iou_avg, train_viola_jones
from utils.data_loader import FileManager

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    coloredlogs.install()
    coloredlogs.set_level(logging.DEBUG)
    logging.info('Started')

    filenames, train_set, test_set, ground_truths = FileManager.prepare_data(base_path='data/', train_ratio=0.01)
    viola_jones_model = train_viola_jones(image_paths=train_set, base_path='data/', ground_truths=ground_truths)

    logging.info('Finished')
