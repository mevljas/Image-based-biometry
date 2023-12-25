from ultralytics import YOLO
import os, itertools
from datetime import datetime

# Function to update the log file
def update_log_file(log_entry, file_name):
    with open(file_name, "a") as f:
        f.write(log_entry + "\n")

version = "2_"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = "training_log_{}_{}.txt".format(version, timestamp)

# Move to the dir of this script        
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

# Load the model, feel free to try other models
model = YOLO("yolov8n.pt")

# All values
""" 
hsv_hs = [0,  0.01, 0.02, 0.03, 0.05] # Small changes in hue
hsv_ss = [0.2, 0.3, 0.5] # Moderate to high changes in saturation
hsv_vs = [0.2, 0.3, 0.5] # Moderate to high changes in value
degreess = [5, 10, 15] #5, 10, 15 degrees rotation
translations = [0.1, 0.2] # 10% to 20% translation
scales = [0.1, 0.2, 0.3] # 10%, 20%, 30% scaling
shears = [0, 10] #Up to 10 degrees shearing
perspectivess = [0, 0.001] # Small perspective warping
flipuds = [0] # No vertical flipping
fliplrs = [0.5] # 50% chance of flipping horizontally
mosaics = [0.5] # 50% chance of applying mosaic augmentation
mixups = [0.0] # No mixup augmentation
copy_pastes = [0.0] # No copy-paste augmentation
optimizer="AdamW"
dropout=0
lr=0.001
weight_decay=0.0001 """

optimizer="AdamW"
dropout=0
lr=0.001
weight_decay=0.0001

hsv_hs = [0] # No changes in hue
hsv_ss = [0.2] # Moderate changes in saturation
hsv_vs = [0.5] # Moderate changes in value
degreess = [5] #5 degrees rotation
translations = [0.2] # 20% translation
scales = [0.1] # 10% scaling
shears = [0] #0 degrees shearing
perspectivess = [0] # No perspective warping
flipuds = [0] # No vertical flipping
fliplrs = [0.5] # 50% chance of flipping horizontally
mosaics = [0.5] # 50% chance of applying mosaic augmentation
mixups = [0.0] # No mixup augmentation
copy_pastes = [0.0] # No copy-paste augmentation



param_combinations = list(itertools.product(hsv_hs, hsv_ss, hsv_vs, degreess, translations, scales, shears, perspectivess, flipuds, fliplrs, mosaics, mixups, copy_pastes))
completed_combinations = 0
total_combinations = len(param_combinations)
# Initialize variables to store the best model
best_map50_95 = -1
best_model_params = {}

# Initialize log file
with open(log_file_name, "w") as f:
    f.write("Training Log\n")

# Dear colleagues, for the full list and explanations, refer to: https://docs.ultralytics.com/usage/cfg/#train
for hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste in param_combinations:
    # Training
    print("Training on the following hyperparameters: hsv_h: {}, hsv_s: {}, hsv_v: {}, degrees: {}, translate: {}, scale: {}, shear: {}, perspective: {}, flipud: {}, fliplr: {}, mosaic: {}, mixup: {}, copy_paste: {}".format(hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste))
    results = model.train(
        data="ears.yaml",
        save=True,
        project="train2",
        name=version,
        epochs=20, 
        optimizer=optimizer, 
        pretrained=True, 
        patience=3,
        plots=False,
        val=True,
        augment=True,
        dropout=dropout,
        lr0=lr,
        lrf=0.2,
        momentum=0.937,
        weight_decay=weight_decay,
        warmup_epochs=3,
        warmup_momentum=0.5,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        hsv_h = hsv_h,
        hsv_s = hsv_s,
        hsv_v = hsv_v,
        degrees = degrees,
        translate = translate,
        scale = scale,
        shear = shear,
        perspective = perspective,
        flipud = flipud,
        fliplr = fliplr,
        mosaic = mosaic,
        mixup = mixup,
        copy_paste = copy_paste
    )
    # Validation
    metrics = model.val()  # Evaluate the data we trained
    map50_95 = metrics.box.map  # mAP50-95
    # Progress
    completed_combinations += 1
    progress = (completed_combinations / total_combinations) * 100
    print("Hyperparameters progress: {:.2f}%".format(progress))
    # Logging
    log_entry = "Model: yolov8n.pt, Ohsv_h: {}, hsv_s: {}, hsv_v: {}, degrees: {}, translate: {}, scale: {}, shear: {}, perspective: {}, flipud: {}, fliplr: {}, mosaic: {}, mixup: {}, copy_paste: {}".format(hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste)
    update_log_file(log_entry, log_file_name)

    # Update best model
    if map50_95 > best_map50_95:
        best_map50_95 = map50_95
        best_model_params = {
            "model": "yolov8n.pt",
            "optimizer": optimizer,
            "lr": lr,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "shear": shear,
            "perspective": perspective,
            "flipud": flipud,
            "fliplr": fliplr,
            "mosaic": mosaic,
            "mixup": mixup,
            "copy_paste": copy_paste,
            "mAP50-95": map50_95,
            "save_dir": results.save_dir
        }

# Print best model details
print("Best Model Parameters:", best_model_params, "with value:", best_map50_95)
