import cv2

class HOGFeatureExtractor:
    def __init__(self, win_size=(64, 64), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def extract_features(self, image):
        # Convert to grayscale if the image is in color
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to meet HOG descriptor requirements
        if image is not None and (image.shape[0] != 128 or image.shape[1] != 64):
            image = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)

        # Ensure the image is in the correct data type
        image = cv2.convertScaleAbs(image)

        # Compute HOG features
        hog_features = self.hog.compute(image).flatten()

        return hog_features
