import logging
from typing import Any

import cv2


def detect_ear(ear_detector: cv2.CascadeClassifier, img_path: str) -> (int, int, int, int, int, int):
    """
    Detects ears on the given image. Returns the bounding box coordinates if an ear is detected.
    :param ear_detector:  cascade classifier for ear detection.
    :param img_path:  path of the image.
    :return:  bounding box coordinates if an ear is detected.
    """
    logging.debug('Detecting ears with ear detector: ' + str(ear_detector))
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = []

    # results is a list of bounding box coordinates (x,y,width,height) around the detected object.
    results = ear_detector.detectMultiScale(gray,
                                            scaleFactor=1.01,
                                            # How much the object’s size is reduced to the original image (1-2).
                                            minNeighbors=5,
                                            # How many neighbors should contribute in a single bounding box.
                                            minSize=(30, 30),  # Minimum possible object size.
                                            )

    for (x, y, width, height) in results:
        logging.info(
            f'Ear detected x: ' + str(x) + ', y: ' + str(y) + ' width: ' + str(width) + ', height: ' + str(height))
        # Draw rectangles after passing the coordinates.
        # cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

        detections.append((x, y, x + width, y + height, img.shape[1], img.shape[0]))

    logging.debug('Detected ' + str(len(detections)) + ' ears.')

    # cv2.rectangle(img, (516, 518), (707, 746), (255, 0, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return detections[0] if len(detections) > 0 else None


def normalise_ground_truth_coordinates(x1: float,
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


def detect_ears(image_paths: [str], base_path: str) -> dict[Any, list[tuple[int, int, int, int, int, int]] | None]:
    """
    Detects ears on the given images. Returns the bounding box coordinates if an ear is detected.
    :param image_paths: file paths of the images.
    :param base_path: base path for cascade files.
    :return: dictionary of detected bounding boxes per images.
    """
    logging.debug('Detecting ears on ' + str(len(image_paths)) + ' images.')

    left_ear_detector = cv2.CascadeClassifier(base_path + 'haarcascade_mcs_leftear.xml')
    right_ear_detector = cv2.CascadeClassifier(base_path + 'haarcascade_mcs_rightear.xml')
    detections = dict()

    for image in image_paths:
        full_image_path = image + '.png'
        logging.debug(f'Detecting ears on image: ' + full_image_path)

        left_ear_detection = detect_ear(ear_detector=left_ear_detector, img_path=full_image_path)
        right_ear_detection = detect_ear(ear_detector=right_ear_detector, img_path=full_image_path)

        squares = []
        for detection in [left_ear_detection, right_ear_detection]:
            if detection is not None:
                x1, y1, x2, y2, img_width, img_height = detection

                squares.append((x1, y1, x2, y2, img_width, img_height))
        detections[image] = squares if len(squares) > 0 else None

    logging.info('Ears detected on ' + str(len(detections.keys())) + ' images.')

    return detections


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
    xb, yb, xb2, yb2, _, _ = predicted_box

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
                    _, _, _, _, img_width, img_height = pred_box
                    x, y, x2, y2 = gt_box
                    converted_gt_box = normalise_ground_truth_coordinates(x1=x,
                                                                          y1=y,
                                                                          x2=x2,
                                                                          y2=y2,
                                                                          image_width=img_width,
                                                                          image_height=img_height)

                    iou = calculate_iou(converted_gt_box, pred_box)
                    best_iou = max(best_iou, iou)

            total_iou += best_iou

    average_iou = total_iou / gt_box_count

    logging.debug('Calculated IOU avg: ' + str(average_iou) + '.')

    return average_iou
