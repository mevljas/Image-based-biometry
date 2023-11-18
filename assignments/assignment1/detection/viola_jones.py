import logging

import cv2
import numpy as np


class ViolaJones(object):
    @staticmethod
    def detect_ear(ear_detector: cv2.CascadeClassifier, img_name: str,
                   scale_factor: float,
                   min_neighbours: int,
                   min_size: (int, int),
                   max_size: (int, int)) -> ((int, int, int, int, int, int), (int, int)):
        """
        Detects ears on the given image. Returns the bounding box coordinates if an ear is detected.
        :param max_size:  Maximum possible object size.
        :param min_size: Minimum possible object size.
        :param min_neighbours: How many neighbors should contribute in a single bounding box.
        :param scale_factor: How much the object’s size is reduced to the original image (1-2).
        :param ear_detector:  cascade classifier for ear detection.
        :param img_name:  name of the image without an extension.
        :return:  bounding box coordinates of ears and image size.
        """
        # logging.debug('Detecting ears with ear detector: ' + str(ear_detector))
        img = cv2.imread('data/ears/' + img_name + '.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detections = []
        image_size = dict()

        # results is a list of bounding box coordinates (x,y,width,height) around the detected object.
        results = ear_detector.detectMultiScale(gray,
                                                scaleFactor=scale_factor,
                                                minNeighbors=min_neighbours,
                                                minSize=(min_size, min_size),
                                                maxSize=(max_size, max_size)
                                                )

        for (x, y, width, height) in results:
            x, y, width, height = int(x), int(y), int(width), int(height)
            logging.debug(
                f'Ear detected x: ' + str(x) + ', y: ' + str(y) + ' width: ' + str(width) + ', height: ' + str(height))
            # Draw rectangles after passing the coordinates.
            # cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

            detections.append((x, y, x + width, y + height))

        logging.debug('Detected ' + str(len(detections)) + ' ears.')

        # cv2.rectangle(img, (516, 518), (707, 746), (255, 0, 0), 2)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return detections, (img.shape[1], img.shape[0])

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
                    ViolaJones.normalise_ground_truth(x1=x, y1=y, x2=x2, y2=y2,
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
    def detect_ears(filenames: [str],
                    base_path: str,
                    scale_factor: float,
                    min_neighbours: int,
                    min_size: (int, int),
                    max_size: (int, int)
                    ) -> (dict[str, list[tuple[int, int, int, int, int, int]] | None], (int, int)):
        """
        Detects ears on the given images. Returns the bounding box coordinates if an ear is detected.
        :param filenames: image names without extensions and identities.
        :param base_path: base path for cascade files.
        :param max_size:  Maximum possible object size.
        :param min_size: Minimum possible object size.
        :param min_neighbours: How many neighbors should contribute in a single bounding box.
        :param scale_factor: How much the object’s size is reduced to the original image (1-2).
        :return: dictionaries of detected bounding boxes per images and image sizes.
        """
        logging.debug('Detecting ears on ' + str(len(filenames)) + ' images.')

        left_ear_detector = cv2.CascadeClassifier(base_path + 'haarcascade_mcs_leftear.xml')
        right_ear_detector = cv2.CascadeClassifier(base_path + 'haarcascade_mcs_rightear.xml')
        detections = dict()
        image_sizes = dict()

        for image in filenames:
            logging.debug(f'Detecting ears on image: ' + image)

            left_ear_detections, image_size = ViolaJones.detect_ear(ear_detector=left_ear_detector, img_name=image,
                                                                    scale_factor=scale_factor,
                                                                    min_neighbours=min_neighbours,
                                                                    min_size=min_size,
                                                                    max_size=max_size)

            right_ear_detections, _ = ViolaJones.detect_ear(ear_detector=right_ear_detector, img_name=image,
                                                            scale_factor=scale_factor,
                                                            min_neighbours=min_neighbours,
                                                            min_size=min_size,
                                                            max_size=max_size)

            image_sizes[image] = image_size

            squares = []
            for detection in left_ear_detections + right_ear_detections:
                if detection is not None:
                    x1, y1, x2, y2 = detection
                    squares.append((x1, y1, x2, y2))

            detections[image] = squares if len(squares) > 0 else None

        logging.debug('Ears detected on ' + str(len(detections.keys())) + ' images.')

        return detections, image_sizes

    @staticmethod
    def calculate_iou(ground_truth_box: [float], predicted_box: [float]) -> float:
        """
        Calculates the IOU for the given ground truth and predicted bounding boxes.
        :param ground_truth_box:
        :param predicted_box:
        :return:
        """

        logging.debug('Calculating IOU for ground truth box: ' + str(ground_truth_box) +
                      ', predicted box: ' + str(predicted_box) + '.')

        xa, ya, xa2, ya2 = ground_truth_box
        xb, yb, xb2, yb2 = predicted_box

        # Calculate the coordinates of the intersection rectangle
        x_intersection = max(xa, xb)
        y_intersection = max(ya, yb)
        x2_intersection = min(xa + xa2, xb + xb2)
        y2_intersection = min(ya + ya2, yb + yb2)

        # Calculate the area of the intersection
        intersection_area = max(0, x2_intersection - x_intersection) * max(0, y2_intersection - y_intersection)

        # Calculate the area of the union
        union_area = (xa2 * ya2) + (xb2 * yb2) - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area

        logging.debug('Calculated IOU: ' + str(iou) + '.')

        return iou

    @staticmethod
    def calculate_iou_avg(ground_truths: {float}, predictions: {float}) -> float:
        """
        Calculates the average IOU for the given ground truths and predictions.
        :param ground_truths: ground truths for all images.
        :param predictions: predictions for all images.
        :return: avg IOU.
        """

        logging.debug('Calculating avg IOU for ' + str(len(ground_truths)) + ' ground truths and '
                      + str(len(predictions)) + ' predictions.')

        total_iou = 0
        gt_box_count = 0

        for filename in ground_truths.keys():
            logging.debug('Calculating IOU for image: ' + filename + '.')
            for gt_box in ground_truths[filename]:
                best_iou = 0
                gt_box_count += 1
                if filename not in predictions.keys():
                    continue

                if predictions[filename] is not None:

                    for pred_box in predictions[filename]:
                        iou = ViolaJones.calculate_iou(gt_box, pred_box)
                        best_iou = max(best_iou, iou)

                total_iou += best_iou

        if total_iou > 0:
            average_iou = total_iou / gt_box_count
        else:
            average_iou = 0

        logging.debug('Calculated IOU avg: ' + str(average_iou) + '.')

        return average_iou

    @staticmethod
    def train(filenames: [str], data_path: str, ground_truths: {str}) -> (
            float, (float, int, int, int),
            dict[str, [(int, int, int, int, int, int)]], dict[str, [(int, int, int, int, int, int)]]
    ):
        """
        Trains the Viola-Jones model with different parameters and find parameters with the highest average iou.
        :param filenames: file names of the images without an extension.
        :param data_path: base path for cascade files.
        :param ground_truths: ground truths for all images.
        :return:
        """

        best_ioi = 0
        best_parameters = None
        best_detections = None
        normalized_ground_truths = None

        # New best IOU: 0.6436800056137328 with parameters: (1.01, 3, 20, 750)

        for scale_factor in np.arange(1.01, 1.02, 0.1):
            # for scale_factor in np.arange(1.01, 1.1, 0.1):
            logging.debug('Trying scale factor: ' + str(scale_factor))
            for min_neighbors in range(3, 5, 1):
                # for min_neighbors in range(3, 4, 1):
                logging.debug('Trying min neighbors: ' + str(min_neighbors))
                for min_size in range(20, 40, 5):
                    # for min_size in range(30, 31, 2):
                    logging.debug('Trying min size: ' + str(min_size))
                    for max_size in range(600, 900, 50):
                        # for max_size in range(550, 600, 50):
                        logging.debug('Trying parameters: scale_factor: '
                                      + str(scale_factor) + ', min_neighbors: '
                                      + str(min_neighbors) + ', min_size: '
                                      + str(min_size) + ', max_size: '
                                      + str(max_size))

                        detections, image_sizes = ViolaJones.detect_ears(filenames=filenames,
                                                                         base_path=data_path,
                                                                         scale_factor=scale_factor,
                                                                         min_neighbours=min_neighbors,
                                                                         min_size=min_size,
                                                                         max_size=max_size)

                        if normalized_ground_truths is None:
                            normalized_ground_truths = ViolaJones.normalise_ground_truths(ground_truths=ground_truths,
                                                                                          image_sizes=image_sizes,
                                                                                          filenames=filenames)

                        avg_ioi = ViolaJones.calculate_iou_avg(predictions=detections,
                                                               ground_truths=normalized_ground_truths)
                        if avg_ioi > best_ioi:
                            best_ioi = avg_ioi
                            best_detections = detections
                            best_parameters = (scale_factor, min_neighbors, min_size, max_size)
                            logging.info(
                                'New best IOU: ' + str(best_ioi) + ' with parameters: ' + str(best_parameters))

        logging.debug('Best IOU: ' + str(best_ioi) + ' with parameters: ' + str(best_parameters))

        return best_ioi, best_parameters, best_detections, normalized_ground_truths

    @staticmethod
    def test(filenames: [str],
             data_path: str,
             ground_truths: {str},
             scale_factor: float,
             min_neighbors: int,
             min_size: int,
             max_size: int) -> (
            float,
            dict[str, [(int, int, int, int, int, int)]], dict[str, [(int, int, int, int, int, int)]]
    ):
        """
        Test the Viola-Jones model with the provided parameters and find calculates average iou.
        :param filenames: file names of the images without an extension.
        :param data_path: base path for cascade files.
        :param ground_truths: ground truths for all images.
        :return: Calculated iou and dictionary of detections.
        """

        logging.debug('Testing with parameters: scale_factor: '
                      + str(scale_factor) + ', min_neighbors: '
                      + str(min_neighbors) + ', min_size: '
                      + str(min_size) + ', max_size: '
                      + str(max_size))

        detections, image_sizes = ViolaJones.detect_ears(filenames=filenames,
                                                         base_path=data_path,
                                                         scale_factor=scale_factor,
                                                         min_neighbours=min_neighbors,
                                                         min_size=min_size,
                                                         max_size=max_size)

        normalized_ground_truths = ViolaJones.normalise_ground_truths(ground_truths=ground_truths,
                                                                      image_sizes=image_sizes,
                                                                      filenames=filenames)

        iou = ViolaJones.calculate_iou_avg(predictions=detections,
                                           ground_truths=normalized_ground_truths)

        logging.debug('IOU: ' + str(iou))

        return iou, detections, normalized_ground_truths
