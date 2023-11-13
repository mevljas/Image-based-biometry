from skimage import feature


class ScikitLocalBinaryPattern(object):
    @staticmethod
    def train_local_binary_pattern(image, radius, n_points, method='uniform'):
        """
        Runs the scikit local binary pattern model with the provided parameters and returns the result.
        :return:
        """

        scikit_features = feature.local_binary_pattern(image, n_points, radius,
                                                       method='uniform' if method else 'default')
