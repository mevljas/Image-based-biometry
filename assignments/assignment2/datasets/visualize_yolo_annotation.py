import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2, os, PIL
import os

# Move to the dir of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def visualize_yolo_annotations(image, annotations):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for annotation in annotations:
        x1, y1, w, h = annotation['bbox']
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def load_annotations(filename, image_width, image_height):
    annotations = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            class_id = int(line[0])
            x, y, w, h = map(float, line[1:])
            x *= image_width
            y *= image_height
            w *= image_width
            h *= image_height

            x -= w / 2
            y -= h / 2
            
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            annotations.append({"class": class_id, "bbox": [x, y, w, h]})
    return annotations

# Example usage:
d = ''
fname = '0501.png'
img_path = os.path.join(d, 'earstest/images', fname)
ann_path = os.path.join(d, 'earstest/labels', os.path.splitext(fname)[0] + '.txt')

img = cv2.imread(img_path)
img = img[:, :, :3] # Remove the alpha channel
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im = PIL.Image.fromarray(img)
width, height = im.size

annotations = load_annotations(ann_path, width, height)

visualize_yolo_annotations(img, annotations)