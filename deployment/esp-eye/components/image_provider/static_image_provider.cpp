#include "static_image_provider.h"
#include "cars_image.h"
#include "no_cars_image.h"
#include <stdio.h>

StaticImageProvider::StaticImageProvider(bool object) :
    m_object(object)
{
}

const uint8_t* StaticImageProvider::get(int size)
{
    if (size != g_cars_image_len) {
        printf("StaticImageProvider error: Argument 'size' must equal %d\r\n", g_cars_image_len);
        return nullptr;
    }

    return m_object ? g_cars_image : g_no_cars_image;
}
