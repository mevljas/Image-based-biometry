import cv2
from skimage import feature


class ScikitLocalBinaryPattern(object):
    @staticmethod
    def run(image, radius, neighbor_points, uniform):
        """
        Runs the scikit local binary pattern model with the provided parameters and returns the result.
        :return:
        """

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = feature.local_binary_pattern(gray_image, P=neighbor_points, R=radius,
                                                method='uniform' if uniform else 'default')
        # Correctly handle 3D and 2D features
        return features.flatten() if features.ndim == 2 else features
