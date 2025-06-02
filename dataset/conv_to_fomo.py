import os
import json
import numpy as np
from PIL import Image
import cv2
import csv

# Configuration
base_dir = "loaf"
output_sizes = [96, 128, 224]
out_dir = "fomo_loaf"
os.makedirs(out_dir, exist_ok=True)
num_classes = 2
cell_size = 8  # FOMO output stride

splits = ["train", "val", "test"]

# Helper to crop image and map center
def convert_labels(image_path, anns, size):
    img = Image.open(image_path).convert("RGB")
    original_w, original_h = img.size

    img = img.resize((size, size))
    W, H = size, size
    grid_W, grid_H = W // cell_size, H // cell_size

    label_map = np.zeros((grid_H, grid_W, num_classes), dtype=np.float32)
    label_map[..., 1] = 1.0  # background by default

    count = 0
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2) / original_w * W
        cy = (y + h / 2) / original_h * H
        grid_x = int(cx) // cell_size
        grid_y = int(cy) // cell_size
        if 0 <= grid_x < grid_W and 0 <= grid_y < grid_H:
            label_map[grid_y, grid_x] = [1.0, 0.0]  # person
            count += 1

    return img, label_map, count

# Main conversion loop
for split in splits:
    image_dir = os.path.join(base_dir, split)
    annotation_path = os.path.join(base_dir, f"annotations/instances_{split}.json")

    with open(annotation_path, 'r') as f:
        coco = json.load(f)

    # Map image_id to file name and group annotations
    image_id_to_name = {img['id']: img['file_name'] for img in coco['images']}
    annots_by_image = {}
    for ann in coco['annotations']:
        if ann['category_id'] != 1:
            continue
        img_id = ann['image_id']
        if img_id not in annots_by_image:
            annots_by_image[img_id] = []
        annots_by_image[img_id].append(ann)

    for size in output_sizes:
        out_img_dir = os.path.join(out_dir, split, f"images_{size}")
        out_lbl_dir = os.path.join(out_dir, split, f"labels_{size}")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        count_csv_path = os.path.join(out_dir, split, f"count_{size}.csv")
        with open(count_csv_path, 'w', newline='') as count_file:
            writer = csv.writer(count_file)
            writer.writerow(['filename', 'count'])

            for img_id, file_name in image_id_to_name.items():
                anns = annots_by_image.get(img_id, [])
                img_path = os.path.join(image_dir, file_name)
                if not os.path.exists(img_path):
                    continue

                try:
                    img, label_map, count = convert_labels(img_path, anns, size)
                    img.save(os.path.join(out_img_dir, file_name))
                    np.save(os.path.join(out_lbl_dir, os.path.splitext(file_name)[0] + ".npy"), label_map)
                    writer.writerow([file_name, count])
                except Exception as e:
                    print(f"[!] Error processing {file_name}: {e}")

        print(f"[✓] Completed {split} at {size}x{size}")

print("[✔] All splits processed.")

