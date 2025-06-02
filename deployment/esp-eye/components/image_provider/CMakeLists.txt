idf_component_register(SRCS
                       camera_image_provider.cpp
                       esp_eye_camera_config.cpp
                       static_image_provider.cpp
                       cars_image.c
                       no_cars_image.c
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES esp32-camera)
