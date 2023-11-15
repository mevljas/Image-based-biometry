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

    filenames, train_set, test_set, identities, train_ground_truths, test_ground_truths = FileManager.prepare_data(
        data_path=data_path,
        train_ratio=0.7)

    if run_type == 'train':
        logging.info('Training VJ...')
        iou, best_parameters, detections, normalized_ground_truths = ViolaJones.train(filenames=train_set,
                                                                                      data_path=data_path,
                                                                                      ground_truths=train_ground_truths)
        logging.info(f'Trained VJ with best IOU: {str(iou)} and parameters: {str(best_parameters)}')

        logging.info('Cropping and saving images...')
        FileManager.save_images(detections=detections,
                                grounds_truths=normalized_ground_truths,
                                save_directory=output_path)
        logging.info('Cropping and saving images done.')

        logging.info('Training LBP with scikit...')
        scikit_lbp_accuracy, scikit_lbp_parameters = LocalBinaryPattern.train_local_binary_pattern(
            data_path=output_path + 'ground_truths/',
            filenames=filenames,
            use_scikit=True)
        logging.info(f'Trained scikit LBP and got accuracy: {scikit_lbp_accuracy} with parameters: ' + str(
            scikit_lbp_parameters))

        logging.info('Training custom LBP...')
        my_lbp_best_accuracy, my_lbp_parameters = LocalBinaryPattern.train_local_binary_pattern(
            data_path=output_path + 'ground_truths/',
            filenames=filenames,
            use_scikit=False)
        logging.info(f'Trained custom LBP and got accuracy: {my_lbp_best_accuracy} with parameters: ' + str(
            my_lbp_parameters))

    elif run_type == 'test':
        scale_factor = 1.01
        min_neighbors = 3
        min_size = 30
        max_size = 550
        logging.info(f'Testing VJ with parameters: scale_factor: {scale_factor}, min_neighbors: {min_neighbors}, '
                     f'min_size: {min_size}, max_size: {max_size}.')

        iou, detections, normalized_ground_truths = ViolaJones.test(filenames=test_set,
                                                                    data_path=data_path,
                                                                    ground_truths=test_ground_truths,
                                                                    scale_factor=scale_factor,
                                                                    min_neighbors=min_neighbors,
                                                                    min_size=min_size,
                                                                    max_size=max_size)

        logging.info(f'Testing VJ finished with accuracy: {iou}.')
        logging.info('Cropping and saving images...')
        FileManager.save_images(detections=detections,
                                grounds_truths=normalized_ground_truths,
                                save_directory=output_path)
        logging.info('Cropping and saving images done.')

        radius = 3
        n_points = 16
        uniform_option = True

        logging.info(f'Testing scikit LBP with parameters: radius: {radius}, n_points: {n_points}.')
        scikit_lbp_accuracy, scikit_lbp_parameters = LocalBinaryPattern.test_local_binary_pattern(
            data_path=output_path + 'ground_truths/',
            filenames=filenames,
            use_scikit=True,
            radius=radius,
            n_points=n_points,
            uniform_option=uniform_option)
        logging.info(f'Testing scikit LBP finished with accuracy: {scikit_lbp_accuracy}.')

        logging.info(f'Testing custom LBP with parameters: radius: {radius}, n_points: {n_points}.')
        my_lbp_best_accuracy, my_lbp_parameters = LocalBinaryPattern.test_local_binary_pattern(
            data_path=output_path + 'ground_truths/',
            filenames=filenames,
            use_scikit=False,
            radius=radius,
            n_points=n_points,
            uniform_option=uniform_option)
        logging.info(f'Testing custom LBP finished with accuracy: {my_lbp_best_accuracy}.')

        logging.info('Testing P2P...')
        P2P_accuracy = PixelToPixel.test(data_path=output_path + 'ground_truths/', filenames=filenames)
        logging.info(f'Testing P2P finished with accuracy: {str(P2P_accuracy)}.')

    else:
        logging.error(f'Wrong program argument {run_type}.')
        exit(1)

    logging.info('Program finished')
