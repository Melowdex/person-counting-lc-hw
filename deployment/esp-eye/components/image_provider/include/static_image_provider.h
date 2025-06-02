#ifndef STATIC_IMAGE_PROVIDER_H
#define STATIC_IMAGE_PROVIDER_H

#include "image_provider.h"

class StaticImageProvider: public ImageProvider
{
public:
    /*
     *  object: if true, the get function returns an image containing the object
     *          if false, the get function returns an image not containing the object
     */
    StaticImageProvider(bool object);

    const uint8_t* get(int size) override;

private:
    bool m_object;
};

#endif /* STATIC_IMAGE_PROVIDER_H */
