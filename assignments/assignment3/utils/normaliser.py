import logging

import cv2


class Normaliser(object):
    @staticmethod
    def get_image_size(img_name: str, images_path: str) -> (int, int):
        """
        Returns the size of the given image.
        :param images_path: path to the images directory
        :param img_name:  name of the image without an extension.
        :return:  image size.
        """
        img = cv2.imread(images_path + img_name + '.png')

        return img.shape[1], img.shape[0]

    @staticmethod
    def normalise_ground_truths(ground_truths: dict, image_sizes: dict, filenames: [str]):
        normalized_ground_truths = dict()
        for filename in filenames:
            gt_boxes = ground_truths[filename]
            normalized_gt_boxes = []
            img_width, img_height = image_sizes[filename]

            for normalized_gt_box in gt_boxes:
                x, y, x2, y2 = normalized_gt_box
                normalized_gt_boxes.append(
                    Normaliser.normalise_ground_truth(x1=x, y1=y, x2=x2, y2=y2,
                                                      image_width=img_width,
                                                      image_height=img_height))

            normalized_ground_truths[filename] = normalized_gt_boxes

        return normalized_ground_truths

    @staticmethod
    def normalise_ground_truth(x1: float,
                               y1: float,
                               x2: float,
                               y2: float,
                               image_width: int,
                               image_height: int) -> (float, float, float, float):
        """
        Normalizes the ground truth coordinates. The coordinates are given as percentages of the image size.
        :param x1: x in percentage of the image width.
        :param y1: y in percentage of the image height.
        :param x2: x + width in percentage of the image width.
        :param y2: y + height in percentage of the image height.
        :param image_width: Full image width in pixels.
        :param image_height: Full image height in pixels.
        :return: Top left and bottom right coordinates of the bounding box.
        """
        logging.debug('Normalizing ground truth coordinates x1: ' + str(x1) +
                      ', y2: ' + str(y1) +
                      ' x2: ' + str(x2) +
                      ', y1: ' + str(y2) +
                      ' image_width: ' + str(image_width) +
                      ', image_height: ' + str(image_height) + '.')

        x_center = float(x1) * image_width
        y_center = float(y1) * image_height
        ground_width = float(x2) * (image_width / 2)
        ground_height = float(y2) * (image_height / 2)

        norm_x1 = int(x_center - ground_width)
        norm_y1 = int(y_center - ground_height)
        norm_x2 = int(x_center + ground_width)
        norm_y2 = int(y_center + ground_height)

        logging.debug('Normalized ground truth coordinates x1: ' + str(norm_x1) +
                      ', y1: ' + str(norm_y1) +
                      ' x2: ' + str(norm_x2) +
                      ', y2: ' + str(norm_y2) + '.')

        return norm_x1, norm_y1, norm_x2, norm_y2

    @staticmethod
    def get_image_sizes(filenames: [str],
                        images_path: str
                        ) -> (dict[str, list[tuple[int, int, int, int, int, int]] | None], (int, int)):
        """
        Returns image sizes.
        :param filenames: image names without extensions and identities.
        :return: image sizes.
        """
        logging.debug('Getting image sizes of ' + str(len(filenames)) + ' images.')

        image_sizes = dict()

        for image in filenames:
            logging.debug(f'Detecting ears on image: ' + image)

            image_size = Normaliser.get_image_size(img_name=image, images_path=images_path)

            image_sizes[image] = image_size

        logging.debug('Got image sizes.')

        return image_sizes

    @staticmethod
    def normalise(filenames: [str],
                  ground_truths: {str},
                  images_path: str) -> (
            float,
            dict[str, [(int, int, int, int, int, int)]], dict[str, [(int, int, int, int, int, int)]]
    ):
        """
        Normalise ground truths.
        :param filenames: file names of the images without an extension.
        :param ground_truths: ground truths for all images.
        :return: Normalised ground truths.
        """

        image_sizes = Normaliser.get_image_sizes(filenames=filenames, images_path=images_path)

        normalized_ground_truths = Normaliser.normalise_ground_truths(ground_truths=ground_truths,
                                                                      image_sizes=image_sizes,
                                                                      filenames=filenames)

        return normalized_ground_truths
