import logging

import coloredlogs

from recognition.pixel import PixelToPixel
from utils.data_loader import FileManager

if __name__ == '__main__':
    coloredlogs.install()
    coloredlogs.set_level(logging.DEBUG)
    logging.info('Started')
    data_path = 'data/'
    output_path = 'output/'

    filenames, train_set, test_set, ground_truths, identities = FileManager.prepare_data(data_path=data_path,
                                                                                         train_ratio=0.1)
    # viola_jones_model = ViolaJones.train_viola_jones(filenames=train_set, data_path=data_path,
    #                                                  ground_truths=ground_truths)
    # _, _, detections, normalized_ground_truths = viola_jones_model
    # FileManager.save_images(detections=detections,
    #                         grounds_truths=normalized_ground_truths,
    #                         save_directory=output_path)

    # LocalBinaryPattern.train_local_binary_pattern(data_path=output_path + 'ground_truths/', identities=identities)
    PixelToPixel.test(data_path=output_path + 'ground_truths/', identities=identities)

    logging.info('Finished')
