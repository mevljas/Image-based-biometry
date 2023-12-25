from ultralytics import YOLO
import os, itertools, datetime

# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Move to the dir of this script        
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

# Load the model, feel free to try other models
model = YOLO("yolov8n.pt")

# learning_rates = [xx, xx, xx]
# dropouts = [xx, xx, xx, xx]
# weight_decays = [xx, xx,xx]

learning_rates = [0.0001]
dropouts = [0.01]
weight_decays = [0.0005]

param_combinations = list(itertools.product(learning_rates, dropouts, weight_decays))

# Dear colleagues, for the full list and explanations, refer to: https://docs.ultralytics.com/usage/cfg/#train
for lr, dropout, wd in param_combinations:
    print(lr, dropout, wd)
    model.train(
        data="ears.yaml", 
        epochs=20, 
        optimizer='SGD', 
        pretrained=True, 
        patience=3,
        plots=False,
        val=True,
        augment=False,
        dropout=dropout,
        lr0=lr,
        lrf=0.2,
        momentum=0.937,
        weight_decay=wd,
        warmup_epochs=3,
        warmup_momentum=0.5,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5
    )
    metrics = model.val()  # It'll automatically evaluate the data you trained. 

