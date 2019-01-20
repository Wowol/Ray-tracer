#include <vector>
#include "camera.h"
#include "image.h"
#include "sphere.h"

Image render(std::vector<Sphere> const &spheres, Camera &camera);
