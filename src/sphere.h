#ifndef SPHERE_HEADER
#define SPHERE_HEADER

#include "ray.h"
#include "vector3.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Sphere {
   public:
    HD Sphere(Vector3 pos, float r) : position(pos), radius(r) {}
    HD float get_radius() { return radius; }

    HD bool hits_ray(const Ray &ray) {
        return ray.distance_to_point(position) <= radius;
    }

   private:
    Vector3 position;
    float radius;
};

#endif
