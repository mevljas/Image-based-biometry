import logging

import coloredlogs

from recognition.resnet import Resnet
from recognition.HOG_feature_extractor import HOGFeatureExtractor
from utils.normaliser import Normaliser
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

    filenames, identities, ground_truths = FileManager.prepare_data(labels_path=labels_path)

    logging.info(f'Normalizing ground truths.')

    normalized_ground_truths = Normaliser.normalise(filenames=filenames,
                                                    ground_truths=ground_truths,
                                                    images_path=images_path)

    radius = 3
    n_points = 24
    uniform = False

    # # Test scikit LBP on ground truths
    # logging.info(
    #     f'Testing scikit LBP on ground truth images with parameters: radius: {radius}, n_points: {n_points}.')
    # scikit_lbp_accuracy, lbp_features, _ = LocalBinaryPattern.test_local_binary_pattern(
    #     data_path=images_path,
    #     filenames=filenames,
    #     radius=radius,
    #     neighbor_points=n_points,
    #     uniform=uniform)
    # logging.info(f'Testing scikit LBP on ground truth images finished with accuracy: {scikit_lbp_accuracy}. \n')
    #
    # Load the pretrained ResNet50 model
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
    #
    # resnet_accuracy, resnet_features, _ = Resnet.test_resnet(data_path=images_path, filenames=filenames, model=feature_extractor)
    #
    # # Convert lists to numpy arrays
    # # resnet_features_array = np.array(lbp_features)
    # # lbp_features_array = np.array(resnet_features)
    #
    # logging.info(f'Testing Resnet on ground truth images finished with accuracy: {resnet_accuracy}. \n')

    # Initialize HOG feature extractor
    # Initialize HOG feature extractor with optimized parameters
    hog_extractor = HOGFeatureExtractor(win_size=(256, 256), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8),
                                        nbins=9)

    # Test HOG on ground truths
    logging.info(f'Testing Resnet HOG.')
    hog_accuracy, hog_features, _ = Resnet.test_resnet_hog(
        data_path=images_path,
        filenames=filenames,
        hog_extractor=hog_extractor,
        resnet_model=model
    )
    logging.info('Finished testing Resnet with HOG features.')
    logging.info(f'Resnet with HOG features accuracy: {hog_accuracy}')


    logging.info('Program finished')
