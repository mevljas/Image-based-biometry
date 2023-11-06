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
        print(f'Ear detected at x: '+str(x)+', y: '+str(y)+', width: +'+str(w)+', height: '+str(h))
        # Draw rectangles after passing the coordinates.
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x, y, w, h = normalize_result(x=x, y=y, width=w, height=h, img_width=img.shape[1], img_height=img.shape[0])
        print(f'Normalized coordinates x: ' + str(x) + ', y: ' + str(y) + ', width: ' + str(w) + ', height: ' + str(h))

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


def normalize_result(x: float, y: float, width: int, height: int, img_width: int, img_height: int) -> (int, int, int, int):
    # Calculate the center of the bounding box
    center_x = x + (width / 2)
    center_y = y + (height / 2)

    # Calculate the normalized coordinates by dividing the center and dimensions by the image width and height
    normalized_x = center_x / img_width
    normalized_y = center_y / img_height
    normalized_width = width / img_width
    normalized_height = height / img_height

    return normalized_x, normalized_y, normalized_width, normalized_height

def detect_ears(image_paths: [str], base_path: str) -> dict[(int, int, int, int)]:
    left_ear_detector = cv2.CascadeClassifier(base_path+'haarcascade_mcs_leftear.xml')
    right_ear_detector = cv2.CascadeClassifier(base_path+'haarcascade_mcs_rightear.xml')
    detections = {}

    for image in image_paths:
        left_ear_detection = detect_ear(ear_detector=left_ear_detector, img_path=image)
        right_ear_detection = detect_ear(ear_detector=right_ear_detector, img_path=image)

        if left_ear_detection is not None:
            detections[image] = left_ear_detection
        elif right_ear_detection is not None:
            detections[image] = right_ear_detection
        break
    return detections
