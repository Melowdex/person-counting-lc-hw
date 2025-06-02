import os
import numpy
import sys
import tensorflow as tf
import yaml 
import pandas as pd
import time

from ModelBuilder import ModelBuilder

DATA_DIR = os.path.expanduser("../../dataset/hc")
OUT_DIR = "out_m"

RES_FILE = "results.yaml"

LEARNING_RATE = 0.001
NUM_EPOCHS = 30

def create_dataset(mode):   #mode = "train" or "val"
    df = pd.read_csv(f"{DATA_DIR}/{mode}/count.csv")
    path = f"{DATA_DIR}/{mode}/images" 
    df['filename'] = df['filename'].apply(lambda x: os.path.join(path , x))

    filenames = df['filename'].values
    count = df['count'].values

    return tf.data.Dataset.from_tensor_slices((filenames, count)), len(filenames)

def get_sample(path, label, res, train=True):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (res, res))

    if train:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.3)

    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.float32)

def write_res(file_path, name, new_metrics):

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            existing_data = yaml.safe_load(file) or {}
    else:
        existing_data = {}

    existing_data[name] = new_metrics

    with open(file_path, 'w') as file:
        yaml.dump(existing_data, file, sort_keys=False)

def main():

    #3 different datasets needed -> 96, 128, 224

    train_ds = []
    test_ds = []

    for i in [96, 128, 224]:
        dataset_train, count_train = create_dataset("train")
        dataset_test, count_test = create_dataset("val")

        train_ds.append((dataset_train
            .shuffle(count_train)            
            .map(lambda f, c: get_sample(f, c, i, train=True), 
                    num_parallel_calls=tf.data.AUTOTUNE)
            .batch(32)                          
            .prefetch(tf.data.AUTOTUNE)))

        test_ds.append((dataset_test           
            .map(lambda f, c: get_sample(f, c, i, train=False), 
                    num_parallel_calls=tf.data.AUTOTUNE)
            .batch(32)                          
            .prefetch(tf.data.AUTOTUNE)))
        

    lr_scheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, verbose=1,
                                                 mode='auto', min_delta=0.0005, cooldown=0, min_lr=0)


    overal_start_time = time.time()
    for iteration in range(5):

        size_to_ds = {"96": [train_ds[0], test_ds[0]],
                    "128": [train_ds[1], test_ds[1]],
                    "224": [train_ds[2], test_ds[2]]}
        
        model_funtions = {"dense": ModelBuilder.build_model_dense, 
                        "gap": ModelBuilder.build_model_gap, 
                        "depth": ModelBuilder.build_model_depth}
        

        for key, model in model_funtions.items():
            for a in [0.35, 0.75]:
                for r in [96, 128, 224]:
                    curr_model = model((r, r, 3), "imagenet", a, 1)

                    params = curr_model.count_params()
                    
                    name = f"fomo_count_{key}_r{str(r)}_a{str(a).replace(".", "")}_{iteration}"
                    checkpoint_filepath = f"{os.path.join(OUT_DIR, name)}.keras"
                    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=False,
                                                                monitor='val_mae',
                                                                mode='min',
                                                                save_best_only=True)

                    curr_model.compile(optimizer="Adam", loss="mae", metrics=['mae', 'mse'])
                    
                    
                    print(f"Start training model: {name}")
                    curr_time = time.time()

                    curr_model.fit(size_to_ds[str(r)][0],
                        epochs=NUM_EPOCHS,
                        validation_data=size_to_ds[str(r)][1],
                        callbacks=[lr_scheduler_callback, model_checkpoint_callback])
                    
                    print(f"Model {name} trained, took: {time.time() - curr_time}")

                    #save the mae, mse and amount of params
                    saved_model = tf.keras.models.load_model(checkpoint_filepath, compile=True)

                    loss, mae, mse = saved_model.evaluate(size_to_ds[str(r)][1])

                    write_res(RES_FILE, name, {"loss": loss, "mae": mae, "mse": mse, "params": params})


    print(f"Done, took: {time.time() - overal_start_time}")


if __name__ == "__main__":
    main()
