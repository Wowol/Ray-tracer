#include <vector>
#include "color.h"
#include "drawable.h"
#include "image.h"
#include "ray.h"
#include "render.h"
#include "sphere.h"
#include "vector3.h"

#include <iostream>

int main() {
    Vector3 ray_position(0.0f, 0.0f, 0.0f);
    Vector3 direction(4.0f, 1.0f, 0.0f);
    Ray ray(ray_position, direction);

    Vector3 point(5.0f, 5.0f, 0.0f);
    Sphere sphere(point, 3.99f);

    std::cout << ray.distance_to_point(point) << std::endl;
    std::cout << sphere.hits_ray(ray) << std::endl;
    // render(std::vector<Drawable *>()).writePNG("out.png");
    return 0;
}
