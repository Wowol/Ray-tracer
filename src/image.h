#ifndef PW_IMAGE_HEADER
#define PW_IMAGE_HEADER

#include "color.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif


class Image {
public:
    Image();
    Image(int width, int height);
    Image(const Image& other);
    ~Image();

    void create(int width, int height);
    void destroy();

    RGBColor& at(int x, int y);
    const RGBColor& at(int x, int y) const;

    RGBColor& operator()(int x, int y) { return at(x, y); }
    const RGBColor& operator()(int x, int y) const { return at(x, y); }

    void clear(const RGBColor& color);

    void writePNG(const char* filename);
    void readPNG(const char* filename);

    int width() const { return width_; }
    int height() const { return height_; }

    Image& operator=(const Image& other);

    RGBColor* raw() { return pixels; }
private:
    RGBColor* pixels;
    int width_, height_;

};

#endif
