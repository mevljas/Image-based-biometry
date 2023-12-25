import torch
import os
import cv2
import PIL
import numpy as np
from ultralytics import YOLO

WEIGHTS = 'models/model2/weights/best.pt'
DEVICE = 'cuda'  # For predictions you can use CPU
OUT_DIR = os.path.join('worst_images')
UNCERTAINTY_THRESHOLD = 0.4
NUM_IMAGES_TO_SAVE = 5  # Number of images to save with least effective detections

class Predict():
    model = None
    device = None

    def __init__(self):
        if DEVICE == 'cpu':
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        os.makedirs(OUT_DIR, exist_ok=True)

        self.model = YOLO(WEIGHTS)

    @torch.no_grad()
    def run(self, file_path=None):

        # Load model
        model = self.model

        # Load image
        img = PIL.Image.open(file_path).convert("RGB")
        basename = os.path.splitext(os.path.basename(file_path))[0]

        # Make prediction
        results = model.predict(img, save=True, save_txt=False, save_conf=True, augment=False)

        # Keep track of confidence scores for each image
        image_confidences = []

        # Draw bounding boxes around ears:
        img_array = np.array(img)
        for i, res in enumerate(results):
            res = res.cpu()
            if len(res.boxes.data) > 0:
                confidence = res.boxes.data.numpy()[0][4]
                confidence = round(confidence.astype(float), 6)

                # Draw bounding box if confidence is above the threshold
                if confidence > UNCERTAINTY_THRESHOLD:
                    pos = res.boxes.xywh.data.numpy()[0]
                    x1, y1, x2, y2 = [int(round(val)) for val in [pos[0] - (pos[2] / 2), pos[1] - (pos[3] / 2),
                                                                  pos[0] + (pos[2] / 2), pos[1] + (pos[3] / 2)]]

                    # Draw a rectangle around the detected ear
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

                image_confidences.append(confidence)

        # Save the image with bounding boxes
        img_with_boxes = PIL.Image.fromarray(img_array)
        img_with_boxes.save(f'{OUT_DIR}/{basename}_with_boxes.png')

        # Save confidence scores for each image
        with open(f'{OUT_DIR}/{basename}_confidence.txt', 'w') as f:
            for confidence in image_confidences:
                f.write(f'{confidence}\n')


        if len(image_confidences) == 0:
            image_confidences.append(0)

        return basename, max(image_confidences)

if __name__ == "__main__":
    predictor = Predict()
    test_dir = "datasets/ears/images/test"

    worst_images = []

    # Automatically test all images in the test directory
    for file_name in os.listdir(test_dir):
        if file_name.endswith(".png"):
            image_path = os.path.join(test_dir, file_name)
            print(f"Testing image: {image_path}")
            worst_images.append(predictor.run(image_path))

    # Output 5 worst images based on confidence
    worst_images = sorted(worst_images, key=lambda x: x[1])[:NUM_IMAGES_TO_SAVE]
    print("5 Worst Images based on Confidence:")
    for i, (image_name, confidence) in enumerate(worst_images):
        print(f"{i + 1}. {image_name} - Confidence: {confidence}")
