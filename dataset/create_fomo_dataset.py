import os
import numpy as np
from PIL import Image


# Settings
output_sizes = [96, 128, 224]
num_classes = 2  # class 0 = person, class 1 = background

def convert_labels(label_path, size):

    W, H = size, size
    grid_W, grid_H = W // 8, H // 8

    # Create background label map: all background
    label_map = np.zeros((grid_H, grid_W, num_classes), dtype=np.float32)
    label_map[..., 1] = 1.0  # set all to background

    # Read annotations
    if not os.path.exists(label_path):
        return label_map

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts[:5])
            # angle = float(parts[5])  # Ignored -> only centers needed
            x_pixel = x_center * W
            y_pixel = y_center * H
            grid_x = int(x_pixel) // 8
            grid_y = int(y_pixel) // 8

            if 0 <= grid_x < grid_W and 0 <= grid_y < grid_H:
                label_map[grid_y, grid_x] = [1.0, 0.0]  # person

    return label_map




# Main conversion
for size in output_sizes:

    for train_val in ["train", "val"]:
        out_label_dir = f"hc/{train_val}/labels_{size}"
        os.makedirs(out_label_dir, exist_ok=True)

        for image_name in os.listdir(f"hc/{train_val}/images"):
            if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            
            label_path = os.path.join(f"hc/{train_val}/labels", os.path.splitext(image_name)[0] + ".txt")

            label_map = convert_labels(label_path, size)
            output_path = os.path.join(out_label_dir, os.path.splitext(image_name)[0] + ".npy")
            np.save(output_path, label_map)

        print(f"[✓] {train_val} Labels generated for {size}x{size}")
