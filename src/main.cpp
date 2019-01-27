#include <random>
#include <vector>
#include "camera.h"
#include "color.h"
#include "image.h"
#include "light.h"
#include "ray.h"
#include "rectangle.h"
#include "render.h"
#include "sphere.h"
#include "vector3.h"

#include <iostream>

void camera_test() {
    std::vector<Sphere> spheres;
    Vector3 position(0.0f, 0.0f, 0.0f);
    Vector3 vector_to_screen(0.0f, 0.0f, 1.0f);
    float width = 1.6f;
    float height = 0.9f;

    Sphere s(Vector3(-1.0f, 1.2f, 5.0f), 0.5f,
             Material(RGBColor(1, 0.5f, 0.5f), 0, 0));

    Camera camera(position, vector_to_screen, width, height);

    Rectangle screen = camera.get_screen();

    printf("%f %f %f %f\n", screen.left_top_point.x, screen.left_top_point.y,
           screen.right_bottom_point.x, screen.right_bottom_point.y);

    Image img(1920, 1080);
    for (int x = 0; x < img.width(); x++) {
        for (int y = 0; y < img.height(); y++) {
            Vector3 point_on_screen = Vector3(
                screen.left_top_point.x + x * screen.width() / img.width(),
                screen.left_top_point.y - y * screen.height() / img.height(),
                screen.left_top_point.z);

            Vector3 direction(position, point_on_screen);

            Ray r(position, direction);

            if (s.hits_ray(r)) {
                img(x, y) = RGBColor(1, 0.5f, 1);
            } else {
                img(x, y) = RGBColor(0, 0, 0);
            }
        }
    }

    img(img.width() / 2, img.height() / 2) = RGBColor(5.0f, 5.0f, 5.0f);
    img.writePNG("out.png");
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

void random_test() {
    std::random_device rd;
    std::mt19937 gen(rd());
    Vector3 camera_position(0.0f, 4.0f, 0.0f);
    Vector3 vector_to_screen(0, 0, 1);

    float width = 1.6f;
    float height = 0.9f;

    Camera camera(camera_position, vector_to_screen, width, height);

    int number_of_spheres = 1000;
    int min_x = -40;
    int max_x = 40;
    int min_y = 0;
    int max_y = 4;
    int min_z = 10;
    int max_z = 200;
    int min_radius = 0.3f;
    int max_radius = 3.0f;

    std::vector<Light> lights = {Light(Vector3(-10, 10, 0.0f), 0.8),
                                 Light(Vector3(0, 100, 300), 0.8)};

    std::uniform_real_distribution<float> random_x(min_x, max_x);
    std::uniform_real_distribution<float> random_y(min_y, max_y);
    std::uniform_real_distribution<float> random_z(min_z, max_z);
    std::uniform_real_distribution<float> random_radius(min_radius, max_radius);

    std::uniform_real_distribution<float> random_0_1(0, 1);

    std::uniform_real_distribution<float> random_0_0_0_5(0, 0.05f);

    std::vector<Sphere> spheres;

    for (int x = 0; x < number_of_spheres; x++) {
        Sphere sphere(Vector3(random_x(gen), random_y(gen), random_z(gen)), random_radius(gen), Material(RGBColor(random_0_1(gen), random_0_1(gen), random_0_1(gen)), random_0_0_0_5(gen), 0));
        bool intersect = false;
        for (int y = 0; y < x; y++) {
            if (spheres[y].get_position().distance(sphere.get_position()) < spheres[y].get_radius() + sphere.get_radius()) {
                intersect = true;
                break;
            }
        }
        if (!intersect) {
            spheres.push_back(sphere);
        } else {
            x--;
        }
    }

    render(spheres, lights, camera).writePNG("out.png");
}

void render_test() {
    Vector3 position(0.0f, 4.0f, 0.0f);
    Vector3 vector_to_screen(0.0f, 0.0f, 1.0f);
    float width = 1.6f;
    float height = 0.9f;

    std::vector<Light> lights = {Light(Vector3(-10, 10, 0.0f), 0.8),
                                 Light(Vector3(0, 0, 20), 0.8)};

    std::vector<Sphere> spheres = {
        Sphere(Vector3(-0.7f, 5, 12.0f), 0.5,
               Material(RGBColor(1, 0, 0), .03, 0)),  // 0 RED
        Sphere(Vector3(1.3f, 4, 12.0f), 0.7,
               Material(RGBColor(0, 0, 1.0f), .03, 0)),  // 1 BLUE
        Sphere(Vector3(-0.7f, 3, 8.0f), 1,
               Material(RGBColor(1, 1, 1), .02, 0)),  // 2 WHITE
        Sphere(Vector3(2.0f, 2, 8.0f), 1,
               Material(RGBColor(1, 1, 0), .04, 0)),  // 3 YELLOW
        Sphere(Vector3(13.0f, 5, 30.0f), 7,
               Material(RGBColor(0, 0, 0), .9, 0))  // 4 MIRROR
    };

    Camera camera(position, vector_to_screen, width, height);

    render(spheres, lights, camera).writePNG("out.png");
}

int main() {
    // camera_test();
    // vector_test();
    // render_test();
    random_test();
    return 0;
}
