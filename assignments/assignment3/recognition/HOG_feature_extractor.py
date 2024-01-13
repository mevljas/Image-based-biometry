import cv2
import numpy as np

class HOGFeatureExtractor:
    def __init__(self, win_size=(256, 256), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def extract_features(self, image):
        # Ensure the image is in the correct data type
        image = cv2.convertScaleAbs(image)

        # Compute HOG features
        hog_features = self.hog.compute(image).flatten()

        return hog_features

