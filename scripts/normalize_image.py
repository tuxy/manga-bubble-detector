import os
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class BBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    
    def to_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        x_min = int((self.x_center - self.width / 2) * img_width)
        y_min = int((self.y_center - self.height / 2) * img_height)
        x_max = int((self.x_center + self.width / 2) * img_width)
        y_max = int((self.y_center + self.height / 2) * img_height)
        return (x_min, y_min, x_max, y_max)

def check_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return (x1_min < x2_max and x1_max > x2_min and 
            y1_min < y2_max and y1_max > y2_min)

def load_yolo_labels(label_path: str) -> List[BBox]:
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append(BBox(
                    class_id=int(parts[0]),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4])
                ))
    return boxes

def cluster_boxes(boxes: List[BBox], img_width: int, img_height: int, 
                 max_tile_size: int = 640, buffer: int = 128) -> List[List[BBox]]:
    coords = [box.to_coords(img_width, img_height) for box in boxes]
    clusters = []
    used = set()
    
    for i in range(len(boxes)):
        if i in used:
            continue
            
        current_cluster = [boxes[i]]
        cluster_coords = [coords[i]]
        used.add(i)
        
        for j in range(len(boxes)):
            if j not in used and any(check_overlap(coords[j], c) for c in cluster_coords):
                current_cluster.append(boxes[j])
                cluster_coords.append(coords[j])
                used.add(j)
        
        x_min = min(c[0] for c in cluster_coords)
        y_min = min(c[1] for c in cluster_coords)
        x_max = max(c[2] for c in cluster_coords)
        y_max = max(c[3] for c in cluster_coords)
        
        for j in range(len(boxes)):
            if j not in used:
                c = coords[j]
                close_enough = (c[0] <= x_max + buffer and c[2] >= x_min - buffer and 
                              c[1] <= y_max + buffer and c[3] >= y_min - buffer)
                temp_width = max(c[2], x_max) - min(c[0], x_min)
                temp_height = max(c[3], y_max) - min(c[1], y_min)
                if close_enough and temp_width <= max_tile_size + buffer and temp_height <= max_tile_size + buffer:
                    current_cluster.append(boxes[j])
                    cluster_coords.append(c)
                    used.add(j)
                    x_min = min(x_min, c[0])
                    y_min = min(y_min, c[1])
                    x_max = max(x_max, c[2])
                    y_max = max(y_max, c[3])
        
        clusters.append(current_cluster)
    
    return clusters

def adjust_to_tile_size(x_min: int, y_min: int, x_max: int, y_max: int, 
                       tile_size: int) -> Tuple[int, int, int, int]:
    width = x_max - x_min
    height = y_max - y_min
    
    if width < tile_size:
        extra = (tile_size - width) // 2
        x_min -= extra
        x_max = x_min + tile_size
    if height < tile_size:
        extra = (tile_size - height) // 2
        y_min -= extra
        y_max = y_min + tile_size
    
    return (x_min, y_min, x_max, y_max)

def process_tile(image, cluster, x_min, y_min, x_max, y_max, 
                img_width, img_height, output_dir, base_name, tile_idx):
    crop = image[y_min:y_max, x_min:x_max]
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    
    crop_name = f"{base_name}_{tile_idx}.jpg"
    label_name = f"{base_name}_{tile_idx}.txt"
    
    cv2.imwrite(os.path.join(output_dir, crop_name), crop)
    
    with open(os.path.join(output_dir, label_name), 'w') as f:
        for box in cluster:
            x_c, y_c, w, h = box.x_center, box.y_center, box.width, box.height
            new_x = ((x_c * img_width) - x_min) / crop_width
            new_y = ((y_c * img_height) - y_min) / crop_height
            new_w = w * img_width / crop_width
            new_h = h * img_height / crop_height
            if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                f.write(f"{box.class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")

def slice_image(image_path: str, label_path: str, tile_size: int = 640, max_size_factor: float = 2.0):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    boxes = load_yolo_labels(label_path)
    
    # Use the same directory as the input image
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    clusters = cluster_boxes(boxes, img_width, img_height, max_tile_size=tile_size)
    
    tile_idx = 0
    for i, cluster in enumerate(clusters):
        coords = [box.to_coords(img_width, img_height) for box in cluster]
        x_min = min(c[0] for c in coords)
        y_min = min(c[1] for c in coords)
        x_max = max(c[2] for c in coords)
        y_max = max(c[3] for c in coords)
        
        width = x_max - x_min
        height = y_max - y_min
        max_allowed = tile_size * max_size_factor
        
        if width > max_allowed or height > max_allowed:
            for y_start in range(y_min, y_max, tile_size):
                for x_start in range(x_min, x_max, tile_size):
                    x_end = min(x_start + tile_size, x_max)
                    y_end = min(y_start + tile_size, y_max)
                    if x_end - x_start < tile_size // 2 or y_end - y_start < tile_size // 2:
                        continue
                    x_start = max(0, x_start)
                    y_start = max(0, y_start)
                    x_end = min(img_width, x_end)
                    y_end = min(img_height, y_end)
                    process_tile(image, cluster, x_start, y_start, x_end, y_end, 
                               img_width, img_height, output_dir, base_name, tile_idx)
                    tile_idx += 1
        else:
            x_min, y_min, x_max, y_max = adjust_to_tile_size(x_min, y_min, x_max, y_max, tile_size)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            process_tile(image, cluster, x_min, y_min, x_max, y_max, 
                        img_width, img_height, output_dir, base_name, tile_idx)
            tile_idx += 1
    
    # Delete original files after successful processing
    try:
        os.remove(image_path)
        os.remove(label_path)
    except OSError as e:
        print(f"Error deleting original files {image_path} or {label_path}: {e}")

# List of images to process
image_list = [
    ...
    ...
]

if __name__ == "__main__":
    tile_size = 640
    max_size_factor = 2.0
    
    for image_path in image_list:
        label_path = os.path.splitext(image_path)[0] + ".txt"
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            continue
            
        print(f"Processing: {image_path}")
        slice_image(image_path, label_path, tile_size=tile_size, max_size_factor=max_size_factor)
        print(f"Finished processing: {image_path}")