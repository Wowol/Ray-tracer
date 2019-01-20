#include <vector>
#include "camera.h"
#include "color.h"
#include "image.h"
#include "ray.h"
#include "rectangle.h"
#include "render.h"
#include "sphere.h"
#include "vector3.h"

#include <iostream>

void camera_test() {
    // int resolution_width = 500;
    // int resolution_height = 300;
    // std::vector<Sphere> spheres;
    // Vector3 position(4.0f, 5.0f, 0.0f);
    // Vector3 vector_to_screen(0.0f, 0.0f, 3.0f);
    // float width = 12.0f;
    // float height = 9.0f;

    // Sphere s(Vector3(4.0f, 5.0f, 14.0f), 4.0f);

    // Camera camera(position, vector_to_screen, width, height);

    // Rectangle screen = camera.get_screen();

    // Image img(500, 300);
    // for (int x = 0; x < resolution_width; x++) {
    //     for (int y = 0; y < resolution_height; y++) {
    //         Vector3 point_on_screen =
    //             Vector3(screen.left_top_point.x + x * screen.width() / resolution_width,
    //                     screen.left_top_point.y - y * screen.height() / resolution_height, screen.left_top_point.z);
    //         Vector3 direction(position, point_on_screen);

    //         Ray r(position, direction);

    //         if (s.hits_ray(r)) {
    //             img(x, y) = RGBColor(1, 0.5f, 1);
    //         } else {
    //             img(x, y) = RGBColor(0, 0, 0);
    //         }
    //     }
    // }
    // img.writePNG("out.png");
}

void vector_test() {
    // Vector3 ray_position(0.0f, 0.0f, 0.0f);
    // Vector3 direction(4.0f, 1.0f, 0.0f);
    // Ray ray(ray_position, direction);

    // Vector3 point(5.0f, 5.0f, 0.0f);
    // Sphere sphere(point, 3.99f);

    // std::cout << ray.distance_to_point(point) << std::endl;
    // std::cout << sphere.hits_ray(ray) << std::endl;
}

void render_test() {
    Vector3 position(4.0f, 5.0f, 0.0f);
    Vector3 vector_to_screen(0.0f, 0.0f, 3.0f);
    float width = 12.0f;
    float height = 9.0f;

    std::vector<Sphere> spheres = {Sphere(Vector3(4.0f, 5.0f, 14.0f), 4.0f), Sphere(Vector3(-8.0f, 4.0f, 12.0f), 4.0f)};

    Camera camera(position, vector_to_screen, width, height);

    render(spheres, camera).writePNG("out.png");
}

int main() {
    // camera_test();
    // vector_test();
    render_test();
    return 0;
}
