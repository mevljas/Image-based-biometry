import cv2

class ORBFeatureExtractor:
    def __init__(self, orb=cv2.ORB_create()):
        self.orb = orb

    def extract_features(self, image):
        # Convert to grayscale if the image is in color
        if image.shape[-1] == 4:  # Check if the image has an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[-1] != 3:  # If not 3 channels and not an alpha channel, something unexpected
            raise ValueError(f"Unexpected number of channels in image: {image.shape[-1]}")

        # Resize the image to meet ORB descriptor requirements
        if image is not None and (image.shape[0] != 128 or image.shape[1] != 64):
            image = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)

        # Ensure the image is in the correct data type
        image = cv2.convertScaleAbs(image)

        # Detect ORB keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)

        # Flatten the descriptors to a 1D array
        orb_features = descriptors.flatten()

        return orb_features
