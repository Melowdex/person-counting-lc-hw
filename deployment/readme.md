### ESP and STM Deployment Instructions

This directory contains code and resources for deploying models on ESP-EYE, ESP32-S3, and STM-based hardware.

#### Directory Structure

* **`common/`**

  * Contains all the used models.
  * Includes an input array of an image.

* **`esp-eye/`**

  * Contains the code and models used on ESP-EYE and ESP32-S3.
  * Deployment is performed using the ESP-IDF framework.
  * The `main.c` file is the same for each model. Minor changes are needed to switch between models:
    * using a define in `models/model_name.cpp`, in the directory of deployment, not the common. Ensure only one define is active at a time.
    * Adjust the input image based on the model name: `r96` or `r128`.
    * Operation resolver for FOMO models:

      ```cpp
      static tflite::MicroMutableOpResolver<4> resolver;
      resolver.AddConv2D();
      resolver.AddDepthwiseConv2D();
      resolver.AddAdd();
      resolver.AddMean();
      ```
    * Operation resolver for MobileNet models:

      ```cpp
      static tflite::MicroMutableOpResolver<6> resolver;
      resolver.AddConv2D();
      resolver.AddDepthwiseConv2D();
      resolver.AddAdd();
      resolver.AddMul();
      resolver.AddFullyConnected();
      resolver.AddReshape();
      ```

* **`stm32/`**

  * Contains the code and configuration files necessary to run the models on STM-based platforms.
  * Deployment is performed using Mbed OS version 6.7.
  * The `main.c` file is the same for each model. Minor changes are needed to switch between models:
    * using a define in `models/model_name.cpp`, in the directory of deployment, not the common. Ensure only one define is active at a time.
    * Adjust the input image based on the model name: `r96` or `r128`.
    * Operation resolver for FOMO models:

      ```cpp
      static tflite::MicroMutableOpResolver<4> resolver;
      resolver.AddConv2D();
      resolver.AddDepthwiseConv2D();
      resolver.AddAdd();
      resolver.AddMean();
      ```
    * Operation resolver for MobileNet models:

      ```cpp
      static tflite::MicroMutableOpResolver<6> resolver;
      resolver.AddConv2D();
      resolver.AddDepthwiseConv2D();
      resolver.AddAdd();
      resolver.AddMul();
      resolver.AddFullyConnected();
      resolver.AddReshape();
      ```

#### Deployment Configuration

**To deploy on ESP-EYE:**

* Enable external SDRAM and allow the `.bss` section to be placed in it.
* SDRAM should be configured in **Quad Mode**.
* Set the CPU clock speed to **240 MHz**.
* Enable custom partitioning, especially required for the MobileNet model.

**To deploy on ESP32-S3:**

* Enable external SDRAM and allow the `.bss` section to be placed in it.
* SDRAM should be configured in **Octal Mode**.
* Set the CPU clock speed to **240 MHz**.
* Enable custom partitioning, especially required for the MobileNet model.

**To deploy on STM (e.g., DISCO\_F746NG):**

* A Python virtual environment with Mbed set up is provided as `.mbed_67`.

* If issues occur, refer to the official Mbed OS setup guide: [Mbed OS 6.7 Installation and Setup](https://os.mbed.com/docs/mbed-os/v6.7/build-tools/install-and-set-up.html)

* Before compiling, run:

  ```bash
  mbed deploy
  ```

* Compile with the following command:

  ```bash
  mbed compile -m DISCO_F746NG -t GCC_ARM
  ```

* Additional options:

  * `-f` : Flash the compiled binary directly to the device
  * `--clean` : Start from a clean build
