import tensorflow as tf
import os
import numpy as np
import yaml
import pandas as pd

class DatasetLoader():
    def __init__(self, input_type : str, output_type : str, resolution : int, output_res : int = None):
        """
        Define the in and output datatype.
        - input_type [str]: choice of ["RGB", "GRAY"]
        - output_type [str]: choice of ["count", "heat", "fomo-obj", "mobilv3"]
        - resolution [int]: Input image resolution
        - output_res [int]: Only when "fomo-obj" is selected
        """

        self.input_type = input_type
        self.output_type = output_type
        self.res = resolution
        self.output_res = output_res

        with open("../config.yaml", 'r') as stream:
            self.config = yaml.safe_load(stream)

    def random_flip_density(self, image, density_map):
        flip = tf.less(tf.random.uniform([]), 0.5)
        image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
        density_map = tf.cond(flip, lambda: tf.image.flip_left_right(density_map), lambda: density_map)
        
        flip_ud = tf.less(tf.random.uniform([]), 0.5)
        image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)
        density_map = tf.cond(flip_ud, lambda: tf.image.flip_up_down(density_map), lambda: density_map)

        return image, density_map
    
    def get_sample(self, path, label, train=True):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)

        if self.input_type == "GRAY":
            image = tf.image.rgb_to_grayscale(image)

        image = tf.image.resize(image, (self.res, self.res))
        image = tf.cast(image, tf.float32) / 255.0

        if train:
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness (you can adjust the max_delta as needed)
            image = tf.image.random_brightness(image, max_delta=0.3)

        if self.output_type == "mobilv3":
            image = tf.keras.applications.mobilenet_v3.preprocess_input(image)

        return image, tf.cast(label, tf.float32)
    
    def _load_npy_file(self, npy_path):
        return np.load(npy_path.decode("utf-8")).astype(np.float32)
    
    def get_sample_fomo(self, path, ann_path, train=True):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)

        if self.input_type == "GRAY":
            image = tf.image.rgb_to_grayscale(image)

        image = tf.image.resize(image, (self.res, self.res))
        image = tf.cast(image, tf.float32) / 255.0

        ann = tf.numpy_function(self._load_npy_file, [ann_path], tf.float32)
        ann.set_shape([self.res // 8, self.res // 8, 2])

        if train:
            image, ann = self.random_flip_density(image, ann)

        if self.output_type == "mobilv3":
            image = tf.keras.applications.mobilenet_v3.preprocess_input(image)

        return image, ann

    def create_dataset_count(self, mode):
        df = pd.read_csv(f"{self.config['paths']['count'][mode]}/{self.config['paths']['count']['img_label'][2]}")
        path = f"{self.config['paths']['count'][mode]}/{self.config['paths']['count']['img_label'][0]}" 
        df['filename'] = df['filename'].apply(lambda x: os.path.join(path , x))
        df['count'] = df['count']

        filenames = df['filename'].values
        count = df['count'].values

        return tf.data.Dataset.from_tensor_slices((filenames, count)), len(filenames)

    
    def create_dataset_obj(self, mode):
        path_img = f"{self.config['paths']['fomo-obj'][mode]}/{self.config['paths']['fomo-obj']['img_label'][0]}"
        path_label = f"{self.config['paths']['fomo-obj'][mode]}/{self.config['paths']['fomo-obj']['img_label'][1]}_{str(self.res)}"

        img_list = [os.path.join(path_img, f) for f in os.listdir(path_img)]
        annotations = [os.path.join(path_label, os.path.basename(f.replace(".jpg", ".npy"))) for f in img_list]

        return tf.data.Dataset.from_tensor_slices((img_list, annotations)), len(img_list)

    def get_dataset(self, mode):
        if self.output_type == "count":
            return self.create_dataset_count(mode)
        elif self.output_type == "heat":
            return self.create_dataset_heat(mode)
        elif self.output_type == "fomo-obj":
            return self.create_dataset_obj(mode)
        else:
            print(self.output_type + " not found!")


