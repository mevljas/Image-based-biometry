import logging

import coloredlogs

from recognition.hog.hog import Hog
from recognition.hog.hog_feature_extractor import HOGFeatureExtractor
from recognition.lbp.local_binary_pattern import LocalBinaryPattern
from recognition.orb.orb import Orb
from recognition.orb.orb_feature_exctractor import ORBFeatureExtractor
from recognition.resnet import Resnet
from utils.data_loader import FileManager
import torch
import torch.nn as nn
import torchvision.models as models

if __name__ == '__main__':
    coloredlogs.install()
    coloredlogs.set_level(logging.INFO)
    logging.info('Program startup.')
    images_path = 'datasets/ears/images-cropped/test/'
    labels_path = 'datasets/ears/labels/test/'

    filenames, identities = FileManager.prepare_data()

    radius = 3
    n_points = 24
    uniform = True

    # Test scikit LBP on ground truths
    logging.info(
        f'Testing scikit LBP with parameters: radius: {radius}, n_points: {n_points}, uniform: {uniform}.')
    scikit_lbp_accuracy, lbp_features, _ = LocalBinaryPattern.test(
        data_path=images_path,
        filenames=filenames,
        radius=radius,
        neighbor_points=n_points,
        uniform=uniform)
    logging.info(f'Testing scikit LBP finished with accuracy: {scikit_lbp_accuracy}. \n')

    # Load the pretrained ResNet50 model
    logging.info(f'Testing ResNet.')
    model = models.resnet50(pretrained=False)

    # Number of classes you want (e.g., 136)
    num_classes = 136

    # Modify the last fully connected layer to match the desired number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the state dictionary from the saved model
    checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))

    # Load the state dictionary into the modified model
    model.load_state_dict(checkpoint, strict=False)

    # Remove the final softmax layer for feature extraction
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    resnet_accuracy, resnet_features, _ = Resnet.test(data_path=images_path, filenames=filenames,
                                                      model=feature_extractor)

    logging.info(f'Testing Resnet finished with accuracy: {resnet_accuracy}. \n')

    # Initialize HOG feature extractor
    # Initialize HOG feature extractor with optimized parameters
    hog_extractor = HOGFeatureExtractor(win_size=(64, 64), block_size=(16, 16), block_stride=(8, 8), cell_size=(4, 4),
                                        nbins=9)

    # Test HOG on ground truths
    logging.info(f'Testing HOG.')
    hog_accuracy, hog_features, _ = Hog.test(
        data_path=images_path,
        filenames=filenames,
        hog_extractor=hog_extractor,
    )
    logging.info('Finished testing HOG with accuracy: {hog_accuracy}')

    # Test ORB on ground truths
    logging.info(f'Testing ORB.')

    # Create an instance of ORBFeatureExtractor
    orb_extractor = ORBFeatureExtractor()

    # Test Resnet with ORB features
    resnet_orb_accuracy, resnet_orb_features, _ = Orb.test(
        data_path=images_path,
        filenames=filenames,
        orb_extractor=orb_extractor
    )

    logging.info('Finished testing ORB with accuracy: {resnet_orb_accuracy}')

    logging.info('Program finished')
