#include <limits>
#include <vector>
#include "camera.h"
#include "color.h"
#include "light.h"
#include "ray.h"
#include "rectangle.h"
#include "render.h"
#include "vector3.h"

#define BACKGROUND_COLOR RGBColor(0, 1, 1);
#define FLOOR_COLOR RGBColor(0, 1, 0);

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

static constexpr int CHUNK_SIZE = 10;
static constexpr int MAX_NUMBER_OF_REFLECTIONS = 30;
static constexpr float FLOAT_INFINITY = std::numeric_limits<float>::infinity();

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

static __device__ bool is_in_shadow(Vector3 const &point, Light const &light,
                                    Sphere *spheres, int const &spheres_count) {
    Vector3 direction(point, light.position);
    direction.normalize();

    Ray ray(point, direction);

    for (int sphere_index = 0; sphere_index < spheres_count; sphere_index++) {
        if (spheres[sphere_index].hits_ray(ray)) {
            if (spheres[sphere_index].get_intersection_point(ray).distance(
                    point) < light.position.distance(point)) {
                return true;
            }
        }
    }

    return false;
}

static __device__ RGBColor get_light_color(Vector3 const &point,
                                            Light const &light,
                                            Vector3 const &surface_normal) {
    Vector3 light_direction = Vector3(point, light.position);
    light_direction.normalize();

    return light.emitting_light * max(0.0f, surface_normal.scalar_product(light_direction));
}

static __device__ RGBColor get_color(int hit_sphere, Ray const &ray,
                                     Sphere *spheres, Light *lights,
                                     int const &spheres_count,
                                     int const &lights_count) {
    RGBColor surface_color(0, 0, 0);

    for (int light_index = 0; light_index < lights_count; light_index++) {
        Light light = lights[light_index];
        Vector3 hit_point = spheres[hit_sphere].get_intersection_point(ray);
        Vector3 offset = Vector3(hit_point, spheres[hit_sphere].get_position());
        offset.normalize();
        bool in_shadow = is_in_shadow(hit_point - offset * 0.001f, light,
                                      spheres, spheres_count);

        if (!in_shadow) {
            Vector3 surface_normal(spheres[hit_sphere].get_position(), hit_point);
            surface_normal.normalize();
            surface_color =
                surface_color + spheres[hit_sphere].get_material().get_color() *
                                    get_light_color(hit_point, light, surface_normal);
        }
    }
    return surface_color;
}

static __device__ RGBColor cast_ray(Ray ray, Sphere *spheres, Light *lights,
                                    int const &spheres_count,
                                    int const &lights_count) {
    int hit_sphere = -1;
    bool was_hit;
    float color_multiplier = 1;
    RGBColor color(0, 0, 0);
    int old_sphere = -1;

    for (int i = 0; i < MAX_NUMBER_OF_REFLECTIONS; i++) {
        float current_distance = FLOAT_INFINITY;
        was_hit = false;

        for (int sphere_index = 0; sphere_index < spheres_count;
             sphere_index++) {
            if (sphere_index != old_sphere &&
                spheres[sphere_index].hits_ray(ray)) {
                was_hit = true;
                Vector3 intersection_point =
                    spheres[sphere_index].get_intersection_point(ray);
                float distance =
                    Vector3(ray.get_position(), intersection_point).length();
                if (distance < current_distance) {
                    current_distance = distance;
                    hit_sphere = sphere_index;
                }
            }
        }

        if (was_hit) {
            color = color + color_multiplier *
                                (1 - spheres[hit_sphere]
                                         .get_material()
                                         .get_reflection_coefficient()) *
                                get_color(hit_sphere, ray, spheres, lights,
                                          spheres_count, lights_count);
            color_multiplier *=
                spheres[hit_sphere].get_material().get_reflection_coefficient();

            ray = spheres[hit_sphere].reflect(ray);
            old_sphere = hit_sphere;

            if (color_multiplier < 0.001f) {
                break;
            }
        } else {
            break;
        }
    }

    if (ray.get_direction().y < 0) {
        float distance = -ray.get_position().y / ray.get_direction().y;
        Vector3 hit_point = ray.get_position() + ray.get_direction() * distance;
        for (int i = 0; i < lights_count; i++) {
            if (!is_in_shadow(hit_point, lights[i], spheres, spheres_count)) {
                color = color + get_light_color(hit_point, lights[i], Vector3(0, 1, 0)) * color_multiplier * FLOOR_COLOR;
            }
        }
    } else {
        color = color + color_multiplier * BACKGROUND_COLOR;
    }

    return color;
}

static __global__ void kernel(int width, int height, RGBColor *img,
                              Sphere *spheres, Light *lights, Camera camera,
                              int spheres_count, int lights_count) {
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidX > width || tidY > height) {
        return;
    }

    img[tidY * width + tidX] = RGBColor(0.0f, 0.0f, 0.0f);

    Rectangle screen = camera.get_screen();

    RGBColor color(0, 0, 0);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            Vector3 point_on_screen =
                Vector3(screen.left_top_point.x + (tidX + x * 0.1f) * screen.width() / width,
                        screen.left_top_point.y - (tidY + y * 0.1f) * screen.height() / height,
                        screen.left_top_point.z);
            Vector3 direction(camera.position, point_on_screen);
            Ray ray(camera.position, direction);
            color = color + cast_ray(ray, spheres, lights, spheres_count, lights_count);
        }
    }
    img[tidY * width + tidX] = color / 9.0f;
}

Image render(std::vector<Sphere> const &spheres,
             std::vector<Light> const &lights, Camera &camera) {
    cudaSetDevice(0);
    Image img(1920, 1080);

    RGBColor *cudaPixels;
    gpuErrchk(
        cudaMalloc(&cudaPixels, sizeof(RGBColor) * img.width() * img.height()));

    Sphere *cudaSpheres;
    gpuErrchk(cudaMalloc(&cudaSpheres, sizeof(Sphere) * spheres.size()));
    gpuErrchk(cudaMemcpy(cudaSpheres, spheres.data(),
                         sizeof(Sphere) * spheres.size(),
                         cudaMemcpyHostToDevice));

    Light *cudaLights;
    gpuErrchk(cudaMalloc(&cudaLights, sizeof(Light) * lights.size()));
    gpuErrchk(cudaMemcpy(cudaLights, lights.data(),
                         sizeof(Light) * lights.size(),
                         cudaMemcpyHostToDevice));

    uint32_t gridX = (img.width() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    uint32_t gridY = (img.height() + CHUNK_SIZE - 1) / CHUNK_SIZE;

    kernel<<<{gridX, gridY}, {CHUNK_SIZE, CHUNK_SIZE}>>>(
        img.width(), img.height(), cudaPixels, cudaSpheres, cudaLights, camera,
        spheres.size(), lights.size());

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(img.raw(), cudaPixels,
                         sizeof(RGBColor) * img.width() * img.height(),
                         cudaMemcpyDeviceToHost));
    cudaFree(cudaPixels);
    cudaFree(cudaSpheres);

    return img;
}
