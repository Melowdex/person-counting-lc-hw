### Preparing the Dataset

To train and test FOMO models, you first need to generate FOMO-compatible annotations.

1. **Generate FOMO annotations:**

   Run the following script:

   ```bash
   python create_fomo_dataset.py
   ```

2. **Generate gaus annotations:**

   Run the following script:

   ```bash
   python conv_to_dens.py
   ```

3. **(Optional) Add the LOAF Dataset:**

   If you want to include the LOAF dataset for training or validation:

   * Download the dataset (images and annotations) from the official [LOAF website](https://loafisheye.github.io/download.html).
   * Extract the contents into a folder named `loaf` in your working directory.
   * Make sure to place the annotation files from the `resolution_512` set into the following path:

     ```
     loaf/
     ├── annotations/
     │   ├── instances_test-seen.json
     │   ├── instances_test-unseen.json
     │   ├── instances_test.json
     │   ├── instances_train.json
     │   ├── instances_val-seen.json
     │   ├── instances_val-unseen.json
     │   └── instances_val.json
     ├── test/
     ├── train/
     └── val/
     ```

3. **Convert to FOMO Format:**

   Once everything is in place, run the following command to convert the dataset, this can take a wile:

   ```bash
   python conv_to_fomo.py
   ```

4. **Cleanup (Optional):**

   After the conversion is complete, the `loaf` directory is no longer needed and may be deleted.
