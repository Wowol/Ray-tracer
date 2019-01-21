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

void camera_test()
{
    std::vector<Sphere> spheres;
    Vector3 position(4.0f, 5.0f, 0.0f);
    Vector3 vector_to_screen(0.0f, 0.0f, 2.0f);
    float width = 8.0f;
    float height = 6.0f;

    Sphere s(Vector3(4.0f, 15.0f, 10.0f), 4.0f);

    Camera camera(position, vector_to_screen, width, height);

    Rectangle screen = camera.get_screen();

    Image img(800, 600);
    for (int x = 0; x < img.width(); x++)
    {
        for (int y = 0; y < img.height(); y++)
        {
            Vector3 point_on_screen =
                Vector3(screen.left_top_point.x + x * screen.width() / img.width(),
                        screen.left_top_point.y - y * screen.height() / img.height(), screen.left_top_point.z);

          //  printf("%f %f %f\n", point_on_screen.x, point_on_screen.y, point_on_screen.z);
            Vector3 direction(position, point_on_screen);

            Ray r(position, direction);

            if (s.hits_ray(r))
            {
                img(x, y) = RGBColor(1, 0.5f, 1);
            }
            else
            {
                img(x, y) = RGBColor(0, 0, 0);
            }
        }
    }

    img(400, 300) = RGBColor(5.0f, 5.0f, 5.0f);
    img.writePNG("out.png");
}

void vector_test()
{
    // Vector3 ray_position(0.0f, 0.0f, 0.0f);
    // Vector3 direction(4.0f, 1.0f, 0.0f);
    // Ray ray(ray_position, direction);

    // Vector3 point(5.0f, 5.0f, 0.0f);
    // Sphere sphere(point, 3.99f);

    // std::cout << ray.distance_to_point(point) << std::endl;
    // std::cout << sphere.hits_ray(ray) << std::endl;
}

void render_test()
{
    Vector3 position(4.0f, 5.0f, 0.0f);
    Vector3 vector_to_screen(0.0f, 0.0f, 3.0f);
    float width = 16.0f;
    float height = 9.0f;

    std::vector<Sphere> spheres = {Sphere(Vector3(20.0f, 5.0f, 14.0f), 4.0f), Sphere(Vector3(-8.0f, 4.0f, 12.0f), 4.0f)};

    Camera camera(position, vector_to_screen, width, height);

    render(spheres, camera).writePNG("out.png");
}

int main()
{
    // camera_test();
    // vector_test();
    render_test();
    return 0;
}
