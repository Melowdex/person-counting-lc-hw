#ifndef CAMERA_IMAGE_PROVIDER_H
#define CAMERA_IMAGE_PROVIDER_H

#include "image_provider.h"

class CameraImageProvider: public ImageProvider
{
public:
    CameraImageProvider();

    const uint8_t* get(int size) override;

private:
    bool m_init;
    uint8_t* m_buffer;
};


#endif /* CAMERA_IMAGE_PROVIDER_H */
