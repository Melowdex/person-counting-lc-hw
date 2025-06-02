### ESP and STM Deployment Instructions

This directory contains code and resources for deploying models on ESP-EYE, ESP32-S3, and STM-based hardware.

#### Directory Structure

* **`common/`**

  * Contains all the used models.
  * Includes an input array of an image.

* **`esp-eye/`**

  * Contains the code and models used on ESP-EYE and ESP32-S3.
  * Deployment is performed using the ESP-IDF framework.

* **`stm/`**

  * Contains the code and configuration files necessary to run the models on STM-based platforms.
  * Deployment is performed using mbed version 6.7.

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
