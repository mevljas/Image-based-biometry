import numpy as np


class CustomLocalBinaryPattern(object):

    @staticmethod
    def get_pixel_value(image, center, x, y):
        """Get pixel value at a given position."""
        try:
            return 1 if np.any(image[y, x] >= center) else 0
        except IndexError:
            return 0

    @staticmethod
    def calculate_lbp_pixel(image, center, x, y, radius, neighbors_points):
        """Calculate LBP value for a pixel."""
        values = []
        for i in range(neighbors_points):
            angle = i * (360 / neighbors_points)
            dx = round(x + radius * np.cos(np.radians(angle)))
            dy = round(y - radius * np.sin(np.radians(angle)))
            values.append(CustomLocalBinaryPattern.get_pixel_value(image, center, dx, dy))
        return values

    @staticmethod
    def is_uniform(lbp_values):
        """Check if LBP values are uniform."""
        transitions = sum((lbp_values[i] != lbp_values[(i + 1) % len(lbp_values)]) for i in range(len(lbp_values)))
        return transitions <= 2

    @staticmethod
    def local_binary_pattern(image, radius, neighbors_points, use_uniform):
        """Compute LBP for each pixel in the image."""
        if len(image.shape) == 3:
            height, width, _ = image.shape
        else:
            height, width = image.shape
        lbp_image = np.zeros((height, width), dtype=np.uint8)

        for y in range(radius, height - radius):
            for x in range(radius, width - radius):
                center = image[y, x]
                lbp_values = CustomLocalBinaryPattern.calculate_lbp_pixel(image, center, x, y, radius, neighbors_points)

                if use_uniform and not CustomLocalBinaryPattern.is_uniform(lbp_values):
                    lbp_image[y, x] = 255  # Non-uniform pattern
                else:
                    lbp_image[y, x] = sum(value * (2 ** i) for i, value in enumerate(lbp_values))

        return lbp_image

    @staticmethod
    def run(gray_image, radius, neighbors_points, use_uniform):
        lbp_result = CustomLocalBinaryPattern.local_binary_pattern(gray_image, radius, neighbors_points, use_uniform)

        return lbp_result.flatten()
