import logging

import cv2
import numpy as np


def detect_ear(ear_detector: cv2.CascadeClassifier, img_path: str) -> (int, int, int, int, int, int):
    """
    Detects ears on the given image. Returns the bounding box coordinates if an ear is detected.
    :param ear_detector:  cascade classifier for ear detection.
    :param img_path:  path of the image.
    :return:  bounding box coordinates if an ear is detected.
    """
    logging.debug('Detecting ears on image: ' + img_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        # cv2.rectangle(img, (x, y), (x + w, y + height), (0, 255, 0), 2)

        return x, y, width, height, img.shape[1], img.shape[0]

    logging.debug('No ear detected.')

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return None


def convert_percentage_to_pixel_coordinates(x_percentage: float,
                                            y_percentage: float,
                                            width_percentage: float,
                                            height_percentage: float,
                                            image_width: int,
                                            image_height: int) -> (float, float, float, float):

    logging.debug('Converting percentage coordinates to pixel coordinates x: ' + str(x_percentage) +
                  ', y: ' + str(y_percentage) +
                  ' width: ' + str(width_percentage) +
                  ', height: ' + str(height_percentage) +
                  ' image_width: ' + str(image_width) +
                  ', image_height: ' + str(image_height) + '.')

    x_center_pixel = x_percentage * image_width
    y_center_pixel = y_percentage * image_height

    width_pixel = width_percentage * image_width
    height_pixel = height_percentage * image_height

    x_pixel = x_center_pixel - (width_pixel / 2)
    y_pixel = y_center_pixel - (height_pixel / 2)

    logging.debug('Converted percentage coordinates to pixel coordinates x: ' + str(x_pixel) +
                  ', y: ' + str(y_pixel) +
                  ' width: ' + str(width_pixel) +
                  ', height: ' + str(height_pixel) + '.')

    return x_pixel, y_pixel, width_pixel, height_pixel


def detect_ears(image_paths: [str], base_path: str) -> dict[str, list[int]]:
    """
    Detects ears on the given images. Returns the bounding box coordinates if an ear is detected.
    :param image_paths: file paths of the images.
    :param base_path: base path for cascade files.
    :return: dictionary of detected bounding boxes per images.
    """
    logging.debug('Detecting ears on '+str(len(image_paths))+' images.')

    left_ear_detector = cv2.CascadeClassifier(base_path + 'haarcascade_mcs_leftear.xml')
    right_ear_detector = cv2.CascadeClassifier(base_path + 'haarcascade_mcs_rightear.xml')
    detections = dict()

    for image in image_paths:
        full_image_path = image + '.png'
        logging.debug(f'Detecting ears on image: ' + full_image_path)

        left_ear_detection = detect_ear(ear_detector=left_ear_detector, img_path=full_image_path)
        right_ear_detection = detect_ear(ear_detector=right_ear_detector, img_path=full_image_path)

        for detection in [left_ear_detection, right_ear_detection]:
            if detection is not None:
                x, y, w, h, img_width, img_height = detection
                detections[image] = [x, y, w, h, img_width, img_height]

        logging.debug('Got '+str(len(detections))+' detections.')

        return detections

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

    x1, y1, w1, h1 = ground_truth_box
    x2, y2, w2, h2, _, _ = predicted_box

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x2_intersection = min(x1 + w1, x2 + w2)
    y2_intersection = min(y1 + h1, y2 + h2)

    # Calculate the area of the intersection
    intersection_area = max(0, x2_intersection - x_intersection) * max(0, y2_intersection - y_intersection)

    # Calculate the area of the union
    union_area = (w1 * h1) + (w2 * h2) - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    logging.debug('Calculated IOU: ' + str(iou) + '.')

    return iou


def calculate_iou_sum(gt_boxes: {float}, predicted_boxes: {float}) -> float:
    """
    Calculates the average IOU for the given ground truth and predicted bounding boxes.
    :param gt_boxes: ground truth bounding boxes.
    :param predicted_boxes: predicted bounding boxes.
    :return: IOU sum.
    """

    logging.debug('Calculating IOU sum for ' + str(len(gt_boxes)) + ' ground truth boxes and '
                  + str(len(predicted_boxes)) + ' predicted boxes.')

    scores = []

    for key in gt_boxes.keys():
        gt_box = gt_boxes[key]
        if key not in predicted_boxes.keys():
            scores.append(0)
            continue
        pred_box = predicted_boxes[key]
        _, _, _, _, img_width, img_height = pred_box
        x, y, w, h = gt_box
        converted_gt_box = convert_percentage_to_pixel_coordinates(x_percentage=x,
                                                                   y_percentage=y,
                                                                   width_percentage=w,
                                                                   height_percentage=h,
                                                                   image_width=img_width,
                                                                   image_height=img_height)
        scores.append(calculate_iou(converted_gt_box, pred_box))

    logging.debug('Calculated IOU sum: ' + str(sum(scores)) + '.')

    return sum(scores)
