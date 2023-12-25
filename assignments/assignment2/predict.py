import torch, os, sys, cv2, PIL, numpy as np
from ultralytics import YOLO
import os


WEIGHTS = 'models/model2/weights/best.pt'
DEVICE = 'cuda' #cuda # For predictions you can use CPU
OUT_DIR = os.path.join('runs', 'detect')
UNCERTAINTY_THRESHOLD = 0.4

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


        # Optional cropping of ears:
        bbox_arr = []
        for i,res in enumerate(results):
            res = res.cpu()
            if len(res.boxes.data) > 0:
                confidence = res.boxes.data.numpy()[0][4]
                confidence = round(confidence.astype(float), 6)
                cla = int(res.boxes.data.numpy()[0][5])
                # label = res.names[cla].upper()

                if confidence > UNCERTAINTY_THRESHOLD:
                    pos = res.boxes.xywh.data.numpy()[0]
                    
                    # From central to regular:
                    x1 = pos[0] - (pos[2] / 2)
                    y1 = pos[1] - (pos[3] / 2)
                    x2 = pos[0] + (pos[2] / 2)
                    y2 = pos[1] + (pos[3] / 2)

                    bbox = np.round([x1, y1, x2, y2]).astype(int)
                    bbox_arr.append(bbox)
                    cropped = PIL.Image.fromarray(np.array(img)[bbox[1]:bbox[3], bbox[0]:bbox[2]])                    
                    cropped.save(f'{OUT_DIR}/{basename}_cropped_{i}.png')

        return bbox_arr

if __name__ == "__main__":
    predictor = Predict()
    print("\n")
    if len(sys.argv) > 1:
        results = predictor.run(sys.argv[1])
    else:
        def_path = "datasets/ears/images/test/0501.png"
        print("Input image path needed! Defaulting to " + def_path) 
        results = predictor.run(def_path)