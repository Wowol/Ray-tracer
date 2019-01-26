#include <vector>
#include "camera.h"
#include "image.h"
#include "sphere.h"
#include "light.h"

Image render(std::vector<Sphere> const &spheres, std::vector<Light> const &lights, Camera &camera);
