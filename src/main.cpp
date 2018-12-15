#include "color.h"
#include "image.h"
#include "convolve.h"

int main() {
    Image img;
    img.readPNG("tcs1.png");
    convolve(img, 3, {1.0f, 0.0f, -1.0f, 1.0f, 0.3f, -1.0f, 1.0f, 0.0f, -1.0f});
    img.writePNG("tcs2.png");
    return 0;
}
