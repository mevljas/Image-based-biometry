# This is a sample Python script.
import numpy as np

from detection.haar_cascade import detect_ears, calc_iou, get_avg_precision_at_iou
from utils.file_manager import FileManager

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    filenames, train_set, test_set, ground_truths = FileManager.prepare_data(base_path='data/')
    detections = detect_ears(image_paths=train_set, base_path='data/')

    ioi = get_avg_precision_at_iou(ground_truths, detections)
    print(f'IOI: ' + str(ioi))
