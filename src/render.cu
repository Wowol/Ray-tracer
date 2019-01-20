#include <vector>
#include "color.h"
#include "camera.h"
#include "render.h"
#include "ray.h"
#include "vector3.h"
#include "rectangle.h"

static constexpr int CHUNK_SIZE = 32;

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static __global__ void kernel(int width, int height, RGBColor *img, Sphere *spheres, Camera camera,
                              int spheres_count) {
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    if (tidX > width || tidY > height) {
        return;
    }
    
    Rectangle screen = camera.get_screen();

    
    Vector3 point_on_screen = Vector3(screen.left_top_point.x + tidX * screen.width() / width,
    screen.left_top_point.y - tidY * screen.height() / height, screen.left_top_point.z);
    Vector3 direction(camera.position, point_on_screen);
    Ray ray(camera.position, direction);
    
    for (int sphere_index = 0; sphere_index < spheres_count; sphere_index++) {
        if (spheres[sphere_index].hits_ray(ray)) {
            img[tidY*width + tidX] = RGBColor(1, 0.5f, 1);
        } 
    }
}

Image render(std::vector<Sphere> const &spheres, Camera &camera) {
    cudaSetDevice(2);
    Image img(1920, 1080);

    RGBColor *cudaPixels;
    gpuErrchk(cudaMalloc(&cudaPixels, sizeof(RGBColor) * img.width() * img.height()));

    Sphere *cudaSpheres;
    gpuErrchk(cudaMalloc(&cudaSpheres, sizeof(Sphere) * spheres.size()));
    gpuErrchk(cudaMemcpy(cudaSpheres, spheres.data(), sizeof(Sphere) * spheres.size(), cudaMemcpyHostToDevice));

    uint32_t gridX = (img.width() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    uint32_t gridY = (img.height() + CHUNK_SIZE - 1) / CHUNK_SIZE;

    kernel<<<{gridX, gridY}, {CHUNK_SIZE, CHUNK_SIZE}>>>(img.width(), img.height(), cudaPixels, cudaSpheres, camera,
                                                         spheres.size());

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(img.raw(), cudaPixels, sizeof(RGBColor) * img.width() * img.height(), cudaMemcpyDeviceToHost));
    cudaFree(cudaPixels);
    cudaFree(cudaSpheres);

    return img;
}
