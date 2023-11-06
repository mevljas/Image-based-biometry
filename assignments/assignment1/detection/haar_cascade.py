from collections import defaultdict
from typing import Type, Dict, Any, List

import cv2
import numpy as np
from copy import deepcopy


def detect_ear(ear_detector: cv2.CascadeClassifier, img_path: str) -> (int, int, int, int):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # results is a list of bounding box coordinates (x,y,w,h) around the detected object.
    results = (ear_detector.
               # This method only works on grayscale pictures.
               detectMultiScale3(gray,
                                 scaleFactor=1.1,  # How much the object’s size is reduced to the original image (1-2).
                                 minNeighbors=2,  # How many neighbors should contribute in a single bounding box.
                                    outputRejectLevels=True,
                                 # Minimum possible object size. Objects smaller than this are ignored.
                                 )
               )
    rects = results[0]
    neighbours = results[1]
    weights = results[2]

    for coords, score in zip(rects, weights):
        (x, y, w, h) = coords
        print(f'Ear detected at x: ' + str(x) + ', y: ' + str(y) + ', width: ' + str(w) + ', height: ' + str(h))
        # Draw rectangles after passing the coordinates.
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x, y, w, h = normalize_result(x=x, y=y, width=w, height=h, img_width=img.shape[1], img_height=img.shape[0])
        print(f'Normalized coordinates x: ' + str(x) + ', y: ' + str(y) + ', width: ' + str(w) + ', height: ' + str(h))
        print("\n")
        return x, y, w, h, score

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return None


def normalize_result(x: float, y: float, width: int, height: int, img_width: int, img_height: int) -> (
        int, int, int, int):
    # Calculate the center of the bounding box
    center_x = x + (width / 2)
    center_y = y + (height / 2)

    # Calculate the normalized coordinates by dividing the center and dimensions by the image width and height
    normalized_x = center_x / img_width
    normalized_y = center_y / img_height
    normalized_width = width / img_width
    normalized_height = height / img_height

    x_bottomright_gt = normalized_x + normalized_width
    y_bottomright_gt = normalized_y + normalized_height

    return normalized_x, normalized_y, x_bottomright_gt, y_bottomright_gt


def detect_ears(image_paths: [str], base_path: str) -> dict[Any, dict[str, list[Any]] | dict[str, list[Any]]]:
    left_ear_detector = cv2.CascadeClassifier(base_path + 'haarcascade_mcs_leftear.xml')
    right_ear_detector = cv2.CascadeClassifier(base_path + 'haarcascade_mcs_rightear.xml')
    detections = dict()
    scores = []

    for image in image_paths:
        full_image_path = image + '.png'
        print(f'Detecting ears on image: ' + full_image_path)

        left_ear_detection = detect_ear(ear_detector=left_ear_detector, img_path=full_image_path)
        right_ear_detection = detect_ear(ear_detector=right_ear_detector, img_path=full_image_path)

        if left_ear_detection is not None:
            x, y, w, h, score = left_ear_detection
            detections[image] = {'boxes': [x, y, w, h], 'scores': [score]}
        elif right_ear_detection is not None:
            x, y, w, h, score = right_ear_detection
            detections[image] = {'boxes': [x, y, w, h], 'scores': [score]}

    return detections


def calc_iou(gt_bbox, pred_bbox):
    """
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    """
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox

    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p,
                             y_bottomright_gt)

    # if the GT bbox and predcited BBox do not overlap then iou=0
    if (x_bottomright_gt < x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox

        return 0.0
    if (
            y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox

        return 0.0
    if (
            x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox

        return 0.0
    if (
            y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox

        return 0.0

    GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
    Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

    x_top_left = np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

    return intersection_area / union_area


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou(gt_box, pred_box)

            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in iou_sort:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for img_id, res in image_results.items():
        true_positive += res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive / (true_positive + false_positive)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = true_positive / (true_positive + false_negative)
        except ZeroDivisionError:
            recall = 0.0
    return (precision, recall)


def get_avg_precision_at_iou(gt_boxes, pred_bb, iou_thr=0.5):
    model_scores = get_model_scores(pred_bb)
    sorted_model_scores = sorted(model_scores.keys())
    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_bb.keys():
        arg_sort = np.argsort(pred_bb[img_id]['scores'])
        pred_bb[img_id]['scores'] = np.array(pred_bb[img_id]['scores'])[arg_sort].tolist()
        pred_bb[img_id]['boxes'] = np.array(pred_bb[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_bb)

    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        print("Mode score : ", model_score_thr)
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores[model_score_thr]
    for img_id in img_ids:

        gt_boxes_img = gt_boxes[img_id]
        box_scores = pred_boxes_pruned[img_id]['scores']
        start_idx = 0
        for score in box_scores:
            if score <= model_score_thr:
                pred_boxes_pruned[img_id]
                start_idx += 1
            else:
                break
                # Remove boxes, scores of lower than threshold scores:
        pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
        pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]
        # Recalculate image results for this image
        print(img_id)
        img_results[img_id] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr=0.5)
        # calculate precision and recall
    prec, rec = calc_precision_recall(img_results)
    precisions.append(prec)
    recalls.append(rec)
    model_thrs.append(model_score_thr)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls > recall_level).flatten()
            prec = max(precisions[args])
            print(recalls, "Recall")
            print(recall_level, "Recall Level")
            print(args, "Args")
            print(prec, "precision")
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}


def get_model_scores(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_score = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_score.keys():
                model_score[score] = [img_id]
            else:
                model_score[score].append(img_id)
    return model_score
