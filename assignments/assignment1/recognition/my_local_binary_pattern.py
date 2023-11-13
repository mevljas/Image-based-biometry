import cv2
import numpy as np


class MyLocalBinaryPattern(object):

    @staticmethod
    def local_binary_pattern(image, P=8, R=1):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize the LBP feature vector
        lbp_vector = np.zeros((gray.shape[0] - 2 * R, gray.shape[1] - 2 * R), dtype=np.uint8)

        # Calculate LBP for each pixel
        for i in range(R, gray.shape[0] - R):
            for j in range(R, gray.shape[1] - R):
                center = gray[i, j]
                binary_code = 0
                for k in range(P):
                    x = int(np.round(i + R * np.cos(2 * np.pi * k / P)))
                    y = int(np.round(j - R * np.sin(2 * np.pi * k / P)))
                    binary_code |= (gray[x, y] >= center) << k
                lbp_vector[i - R, j - R] = binary_code

        return lbp_vector.flatten()
