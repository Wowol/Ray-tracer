#include "color.h"
#include "render.h"
#include <vector>

static constexpr int CHUNK_SIZE = 32;


#define gpuErrchk(ans)                                                         \
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

static __global__ void kernel(int width, int height, RGBColor *img, Sphere *spheres) { }


Image render(std::vector<Sphere> const &spheres) {
    Image img(1024, 768);
    

    RGBColor *cudaPixels;
    gpuErrchk(cudaMalloc(&cudaPixels, sizeof(RGBColor) * img.width() * img.height()));

    Sphere *cudaSpheres;
    gpuErrchk(cudaMalloc(&cudaSpheres, sizeof(Sphere) * spheres.size()));
    gpuErrchk(cudaMemcpy(cudaSpheres, spheres.data(), sizeof(Sphere) * spheres.size(),
        cudaMemcpyHostToDevice));

    uint32_t gridX = (img.width() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    uint32_t gridY = (img.height() + CHUNK_SIZE - 1) / CHUNK_SIZE;

    kernel<<<{gridX, gridY}, {CHUNK_SIZE, CHUNK_SIZE}>>>(img.width(), img.height(), cudaPixels, cudaSpheres);

    gpuErrchk(cudaMemcpy(img.raw(), cudaPixels,
                         sizeof(RGBColor) * img.width() * img.height(),
                         cudaMemcpyDeviceToHost));
    cudaFree(cudaPixels);
    cudaFree(cudaSpheres);

    return img;
}
