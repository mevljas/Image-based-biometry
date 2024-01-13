import logging

import coloredlogs

from detection.viola_jones import ViolaJones
from recognition.local_binary_pattern import LocalBinaryPattern
from utils.data_loader import FileManager

if __name__ == '__main__':
    coloredlogs.install()
    coloredlogs.set_level(logging.INFO)
    logging.info('Program startup.')
    images_path = 'datasets/ears/images-cropped/test/'
    labels_path = 'datasets/ears/labels/test/'

    filenames, identities, ground_truths = FileManager.prepare_data(labels_path=labels_path)

    logging.info(f'Normalizing ground truths.')

    normalized_ground_truths = ViolaJones.normalise(filenames=filenames,
                                                    ground_truths=ground_truths,
                                                    images_path=images_path)

    radius = 3
    n_points = 24
    uniform = False

    # Test scikit LBP on ground truths
    logging.info(
        f'Testing scikit LBP on ground truth images with parameters: radius: {radius}, n_points: {n_points}.')
    scikit_lbp_accuracy, scikit_lbp_histograms, scikit_lbp_files = LocalBinaryPattern.test_local_binary_pattern(
        data_path=images_path,
        filenames=filenames,
        radius=radius,
        neighbor_points=n_points,
        uniform=uniform)
    logging.info(f'Testing scikit LBP on ground truth images finished with accuracy: {scikit_lbp_accuracy}. \n')

    logging.info('Program finished')
