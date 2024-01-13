import cv2
import numpy as np


class ORBFeatureExtractor:
    def __init__(self, n_keypoints=500):
        self.orb = cv2.ORB_create(n_keypoints)

    def extract_features(self, image):
        # Convert to grayscale if the image is in color
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image if needed
        if image is not None and (image.shape[0] != 128 or image.shape[1] != 64):
            image = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)

        # Compute ORB features
        keypoints, descriptors = self.orb.detectAndCompute(image, None)

        if descriptors is not None and descriptors.shape[0] > 0:
            # Flatten the descriptors
            orb_features = descriptors.flatten()
        else:
            # No keypoints or descriptors found, return an empty array
            orb_features = np.array([])

        return orb_features