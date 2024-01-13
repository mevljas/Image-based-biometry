import cv2
import numpy as np


class ORBFeatureExtractor:
    def __init__(self, ):
        self.orb = cv2.ORB_create(
            nfeatures=2000,    # Adjust based on the number of keypoints you want
            scaleFactor=1.2,   # Adjust the scale factor (e.g., 1.2, 1.4)
            nlevels=8,         # Adjust the number of pyramid levels
            edgeThreshold=31,  # Adjust the edge threshold
            firstLevel=0       # Adjust the first pyramid level
        )

    def extract_features(self, image):
        # Convert to grayscale if the image is in color
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image if needed
        if image is not None and (image.shape[0] != 64 or image.shape[1] != 64):
            image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

        # Compute ORB features
        keypoints, descriptors = self.orb.detectAndCompute(image, None)

        if descriptors is not None and descriptors.shape[0] > 0:
            # Flatten the descriptors
            orb_features = descriptors.flatten()
        else:
            # No keypoints or descriptors found, return an empty array
            orb_features = np.array([])

        return orb_features