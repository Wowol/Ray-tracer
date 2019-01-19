#include "color.h"
#include "render.h"
#include <vector>

Image render(std::vector<Sphere> const &spheres) {
    Image img(1024, 768);
    for (int i = 0; i < img.width(); i++) {
        for (int j = 0; j < img.height(); j++) {
            img(i, j) = RGBColor(1, 0.5, 1);
        }
    }
    return img;
}