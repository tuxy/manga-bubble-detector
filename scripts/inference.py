import os
import cv2
import time
import argparse
from ultralytics import YOLO
from PIL import Image

def run_inference(weight_path, img_folder, output_folder="output"):
    model = YOLO(weight_path)

    os.makedirs(output_folder, exist_ok=True)

    start_time = time.time()

    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)

        if not img_path.lower().endswith((".webp", ".jpg", ".png")):
            continue

        try:
            im = Image.open(img_path)
            width, height = im.size

            height = min(height, 1920) if height > 1600 else 640

            results = model(img_path, imgsz=height)

            for result in results:
                img_with_boxes = result.plot()
                output_path = os.path.join(output_folder, f"output_{img_name}")
                cv2.imwrite(output_path, img_with_boxes)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nInference completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Inference on a Folder")
    parser.add_argument("--weight", type=str, required=True, help="Path to the model weights file (.pt)")
    parser.add_argument("--img_folder", type=str, required=True, help="Folder containing images for inference")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to save processed images")

    args = parser.parse_args()
    run_inference(args.weight, args.img_folder, args.output_folder)
