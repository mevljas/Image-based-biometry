import logging
import sys

import coloredlogs

from detection.viola_jones import ViolaJones
from recognition.local_binary_pattern import LocalBinaryPattern
from recognition.pixel import PixelToPixel
from utils.data_loader import FileManager

if __name__ == '__main__':
    coloredlogs.install()
    coloredlogs.set_level(logging.INFO)
    logging.info('Program startup.')
    data_path = 'data/'
    output_path = 'output/'

    if len(sys.argv) != 2:
        logging.error('Please provide program run type argument (for example train).')
        exit(1)

    run_type = sys.argv[1]

    logging.info('Run type: ' + run_type)

    filenames, train_set, test_set, ground_truths, identities = FileManager.prepare_data(data_path=data_path,
                                                                                         train_ratio=0.7)

    if run_type == 'train':

        _, _, detections, normalized_ground_truths = ViolaJones.train(filenames=train_set,
                                                                      data_path=data_path,
                                                                      ground_truths=ground_truths)

        FileManager.save_images(detections=detections,
                                grounds_truths=normalized_ground_truths,
                                save_directory=output_path)

        scikit_lbp_parameters = LocalBinaryPattern.train_local_binary_pattern(data_path=output_path + 'ground_truths/',
                                                                              identities=identities,
                                                                              use_scikit=True)

        my_lbp_parameters = LocalBinaryPattern.train_local_binary_pattern(data_path=output_path + 'ground_truths/',
                                                                          identities=identities,
                                                                          use_scikit=False)

    elif run_type == 'test':
        iou, detections, normalized_ground_truths = ViolaJones.test(filenames=test_set,
                                                                    data_path=data_path,
                                                                    ground_truths=ground_truths)

        radius = 3
        n_points = 16
        uniform_option = True

        scikit_lbp_parameters = LocalBinaryPattern.test_local_binary_pattern(data_path=output_path + 'ground_truths/',
                                                                             identities=identities,
                                                                             use_scikit=True,
                                                                             radius=radius,
                                                                             n_points=n_points,
                                                                             uniform_option=uniform_option)

        my_lbp_parameters = LocalBinaryPattern.test_local_binary_pattern(data_path=output_path + 'ground_truths/',
                                                                         identities=identities,
                                                                         use_scikit=False,
                                                                         radius=radius,
                                                                         n_points=n_points,
                                                                         uniform_option=uniform_option)

        PixelToPixel.test(data_path=output_path + 'ground_truths/', identities=identities)

    else:
        logging.error(f'Wrong program argument {run_type}.')
        exit(1)

    logging.info('Program finished')
