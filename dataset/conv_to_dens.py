import cv2
import numpy as np
import os
import random
import shutil

import numpy as np

def create_gaussian_map(image_shape, bbox, sigma_scale=1.0):
    """
    Create a rotated anisotropic Gaussian map for a given bounding box.
    
    :param image_shape: Tuple (height, width) of the image.
    :param bbox: Bounding box in YOLO format (x_center, y_center, width, height, angle in degrees).
    :param sigma_scale: Multiplier for sigma_x and sigma_y.
    :return: Gaussian map as a numpy array.
    """
    height, width = image_shape
    x_center, y_center, box_width, box_height, angle = bbox

    x_center = x_center * width
    y_center = y_center * height
    box_width = box_width * width
    box_height = box_height * height

    sigma_x = sigma_scale * box_width / 6  # divide by 6 to get good spread
    sigma_y = sigma_scale * box_height / 6
    theta = np.deg2rad(angle)

    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Shift grid to center
    x_shifted = x_grid - x_center
    y_shifted = y_grid - y_center

    x_rot = np.cos(theta) * x_shifted + np.sin(theta) * y_shifted
    y_rot = -np.sin(theta) * x_shifted + np.cos(theta) * y_shifted

    gaussian_map = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))

    gaussian_map /= np.sum(gaussian_map)

    return gaussian_map


def process_yolo_annotations(img, image_path, annotation_path, output_path, size=96, save=True):
    """
    Process YOLO annotations to create Gaussian maps for each image.
    
    :param image_path: Path to the image file.
    :param annotation_path: Path to the YOLO annotation file.
    :param output_path: Path to save the Gaussian map.
    :param sigma: Standard deviation for the Gaussian kernel.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.resize(image, (size, size))
    height, width, _ = image.shape

    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    # Create an empty Gaussian map
    gaussian_map = np.zeros((height, width), dtype=np.float32)

    # Process each bounding box
    for annotation in annotations:
        parts = annotation.strip().split()
        _, x_center, y_center, box_width, box_height, angle = map(float, parts)
        bbox = (x_center, y_center, box_width, box_height, int(angle))
        gaussian_map += create_gaussian_map((height, width), bbox)

    if save:
        np.save(f"{output_path}/{img.replace('.jpg', '.npy')}", gaussian_map)
    
    return gaussian_map


def get_img_and_anotation(path, img_name, end_res=(96,96)):
    image = cv2.imread(f"{path}/images/{img_name}")
    if image is None:
        print(f"{path}/images/{img_name}")
        raise Exception
    image = cv2.resize(image, end_res)

    with open(f"{path}/labels/{img_name.replace('.jpg', '.txt')}", 'r') as f:
        annotations = f.readlines()

    x = []

    for i in annotations:
        y = dict()
        i = i.strip().split()
        scale_except_last = lambda i: list(map(lambda x: float(x) * end_res[0], i[:-1])) + [float(i[-1])]
        _, y['x'], y['y'], y['width'], y['height'], angle = scale_except_last(i)
        y['angle'] = np.deg2rad(angle)
        x.append(y)

    return image, x

def get_single_gaussian_map(annotations, size=(96,96), sigma_scale=1.0):
    s_x = sigma_scale * annotations['width'] / 2
    s_y = sigma_scale * annotations['height'] / 2

    y_grid, x_grid = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')

    # Shift grid to center
    x_shifted = x_grid - annotations['x']
    y_shifted = y_grid - annotations['y']

    # Rotate grid coordinates
    x_rot = np.cos(annotations['angle']) * x_shifted + np.sin(annotations['angle']) * y_shifted
    y_rot = -np.sin(annotations['angle']) * x_shifted + np.cos(annotations['angle']) * y_shifted

    gaussian_map = np.exp(-(x_rot**2 / (2 * s_x**2) + y_rot**2 / (2 * s_y**2)))

    gaussian_map /= np.sum(gaussian_map)

    return gaussian_map

def get_gaussian_map(annotations, size=(96,96), sigma_scale=1.0):
    tot = np.zeros(size, np.float32)

    for i in annotations:
        tot += get_single_gaussian_map(i, size=size, sigma_scale=sigma_scale)

    return tot

def cut_ppl(img, gaus, center, size=30):
    x, y = int(center[0]), int(center[1])
    half = size // 2

    # Pad the image and gaussian with edge values to ensure enough context
    pad_width = ((half, half), (half, half), (0, 0)) if img.ndim == 3 else ((half, half), (half, half))
    img_padded = np.pad(img, pad_width, mode='edge')
    gaus_padded = np.pad(gaus, ((half, half), (half, half)), mode='edge')

    # Adjust center because of padding
    x_pad = x + half
    y_pad = y + half

    # Crop the patch from padded image
    img_patch = img_padded[y_pad - half:y_pad + half, x_pad - half:x_pad + half]
    gaus_patch = gaus_padded[y_pad - half:y_pad + half, x_pad - half:x_pad + half]

    if img_patch.shape[:2] != (size, size):
        print(img_patch.shape[:2])
        raise Exception
    
    return img_patch, gaus_patch

def save_img_and_gaus(path, name, imgage, gaus):
    if not os.path.exists(f"{path}/images_person"):
        os.mkdir(f"{path}/images_person")
        
    if not os.path.exists(f"{path}/labels_gaus_cut"):
        os.mkdir(f"{path}/labels_gaus_cut")
        
    cv2.imwrite(f"{path}/images_person/{name}.jpg", imgage)
    np.save(f"{path}/labels_gaus_cut/{name}.npy", gaus)

def process(in_dirs, out_dirs, crop_peeps=True):

    for l in range(len(in_dirs)):
        ind = 0
        for i in os.listdir(f"{in_dirs[l]}/images"):
            name, _ = os.path.splitext(i)
        
            
            # save_img_and_gaus(out_dirs[l], name, img, res)
            #per ten images, also save the cropped persons
            if crop_peeps:
                if ind%10 == 0:  
                    img, a = get_img_and_anotation(in_dirs[l], i, end_res=(128,128))
                    res = get_gaussian_map(a, (128,128))  
                    #print("crop_path: " + f"{in_dirs[l]}/images/" + i)                
                    for j in range(len(a)):
                        im, ga = cut_ppl(img, res, (a[j]['x'],a[j]['y']), size=30)
                        save_img_and_gaus(out_dirs[l], f"cut{j:02d}_{name}", im, ga) 
            ind += 1 


if __name__ == "__main__":

    for i in ["hc/train", "hc/val"]:
        image_dir = f"{i}/images"
        annotation_dir = f"{i}/labels"
        output_path = f"{i}/labels_gaus"

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for image in os.listdir(image_dir):
            name, _ = os.path.splitext(image)
            process_yolo_annotations(image, f"{image_dir}/{image}", f"{annotation_dir}/{name}.txt", output_path, size=128)

        print(f"Saved at: {output_path}")

    #next cut some people out.
    print("Start to cut some people out of the images")
    process(["hc/val", "hc/train"], ["hc/val", "hc/train"])

    #create a test_show:
    print("creating a test_show")
    test_show = random.sample(os.listdir("hc/val/images"), 5)

    if not os.path.exists("hc/val/test_show"):
        os.mkdir("hc/val/test_show")

    for i in test_show:
        shutil.copy(f"hc/val/images/{i}", f"hc/val/test_show/{i}")

    test_show = random.sample(os.listdir("hc/val/images_person"), 5)

    if not os.path.exists("hc/val/test_show_person"):
        os.mkdir("hc/val/test_show_person")

    for i in test_show:
        shutil.copy(f"hc/val/images_person/{i}", f"hc/val/test_show_person/{i}")