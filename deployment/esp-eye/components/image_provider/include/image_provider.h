#ifndef IMAGE_PROVIDER_H
#define IMAGE_PROVIDER_H

#include <stdint.h>

class ImageProvider
{
public:
    /*
     *  Get an image
     *  len:    Image size to retrieve (number of uint8's)
     *  returns a pointer to an allocated image buffer of size `size`. Returns nullptr if get failed.
     */
    virtual const uint8_t* get(int size) = 0;
};

#endif /* IMAGE_PROVIDER_H */
