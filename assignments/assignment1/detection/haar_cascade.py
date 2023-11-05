import cv2


def detect_ear(ear_detector: cv2.CascadeClassifier, img_path: str) -> (int, int, int, int):
    img = cv2.imread(img_path + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # results is a list of bounding box coordinates (x,y,w,h) around the detected object.
    results = (ear_detector.
               # This method only works on grayscale pictures.
               detectMultiScale(gray,
                                scaleFactor=1.1,  # How much the object’s size is reduced to the original image (1-2).
                                minNeighbors=2,  # How many neighbors should contribute in a single bounding box.
                                minSize=(30, 30)  # Minimum possible object size. Objects smaller than this are ignored.
                                )
               )
    for (x, y, w, h) in results:
        print('Ear detected at: ', x, y, w, h)
        # Draw rectangles after passing the coordinates.
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return results


def detect_ears(image_paths: [str]) -> dict[(int, int, int, int)]:
    left_ear_detector = cv2.CascadeClassifier('data/ears/haarcascade_mcs_leftear.xml')
    right_ear_detector = cv2.CascadeClassifier('data/ears/haarcascade_mcs_rightear.xml')
    detections = {}

    for image in image_paths:
        left_ear_detection = detect_ear(ear_detector=left_ear_detector, img_path=image)
        right_ear_detection = detect_ear(ear_detector=right_ear_detector, img_path=image)

        if left_ear_detection is not None:
            detections[image] = left_ear_detection
        elif right_ear_detection is not None:
            detections[image] = right_ear_detection

    return detections
