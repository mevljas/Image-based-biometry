import numpy as np


class CustomLocalBinaryPattern(object):

    @staticmethod
    def generate_uniform_patterns():
        patterns = [0] * 256
        for i in range(256):
            binary = bin(i)[2:].zfill(8)
            transitions = sum((int(binary[j]) != int(binary[(j + 1) % 8])) for j in range(8))
            if transitions <= 2:
                patterns[i] = True
        return patterns

    @staticmethod
    def compute_lbp_pixel(img, center, x, y, radius, num_neighbors):
        pattern = 0
        values = []
        for i in range(num_neighbors):
            nx = x + int(round(radius * np.cos(2.0 * np.pi * i / num_neighbors)))
            ny = y - int(round(radius * np.sin(2.0 * np.pi * i / num_neighbors)))

            if nx >= 0 and ny >= 0 and nx < img.shape[1] and ny < img.shape[0]:
                values.append(img[ny, nx])

            if np.any(img[ny, nx] >= center):
                pattern |= (1 << i)

        return pattern, values

    @staticmethod
    def run(img, radius, neighbors_points, use_uniform=True):
        patterns = CustomLocalBinaryPattern.generate_uniform_patterns() if use_uniform else None
        lbp_img = np.zeros_like(img, dtype=np.uint8)

        for y in range(radius, img.shape[0] - radius):
            for x in range(radius, img.shape[1] - radius):
                center = img[y, x]
                pattern, _ = CustomLocalBinaryPattern.compute_lbp_pixel(img, center, x, y, radius, neighbors_points)

                if use_uniform:
                    pattern = patterns[pattern]

                lbp_img[y, x] = pattern

        return lbp_img.flatten()
