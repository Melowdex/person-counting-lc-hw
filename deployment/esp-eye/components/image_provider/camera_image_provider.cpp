#include "camera_image_provider.h"
#include "esp_eye_camera_config.h"

CameraImageProvider::CameraImageProvider() :
    m_init(false), m_buffer(nullptr)
{
}

const uint8_t* CameraImageProvider::get(int size)
{
    if (!m_init) {
        if (!esp_eye_camera_init()) {
            printf("Camera error: initialize failed\r\n");
            return nullptr;
        }
        m_init = true;
    }

    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) {
        printf("Camera error: failed to capture image\r\n");
        return nullptr;
    }

    const int image_size = frame->width * frame->height * 3;
    if (image_size != size) {
        printf("Camera error: image size (%d) is different from expected size (%d)\r\n", image_size, size);
        esp_camera_fb_return(frame);
        return nullptr;
    }

    // convert to RGB888
    if (!m_buffer) {
        m_buffer = static_cast<uint8_t*>(malloc(size));
    }

    fmt2rgb888(frame->buf, frame->len, PIXFORMAT_RGB565, m_buffer);

    esp_camera_fb_return(frame);

    return m_buffer;
}
