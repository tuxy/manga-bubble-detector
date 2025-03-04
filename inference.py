import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

model = YOLO("backup\\balloon_training\weights")

img_folder = "PATH_IMG"

for img_name in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_name)

    if not img_path.lower().endswith((".webp",".jpg",".png")):
        continue

    try:
        im = Image.open(img_path)
        width, height = im.size

        if height > 1600:
            height = min(height, 1920)
        else:
            height = 640

        results = model(img_path, imgsz=height)

        for result in results:
            img_with_boxes = result.plot()

            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(f"{img_name}")
            plt.show()
