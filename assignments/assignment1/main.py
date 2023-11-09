# This is a sample Python script.
import logging

import coloredlogs

from detection.haar_cascade import detect_ears, calculate_iou_avg
from utils.data_loader import FileManager

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    coloredlogs.install()
    coloredlogs.set_level(logging.DEBUG)
    logging.info('Started')

    filenames, train_set, test_set, ground_truths = FileManager.prepare_data(base_path='data/')
    detections = detect_ears(image_paths=train_set, base_path='data/')

    logging.info('IOU avg: ' + str(calculate_iou_avg(predictions=detections, ground_truths=ground_truths)))
    logging.info('Finished')
