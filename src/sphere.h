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
    HD Sphere(Vector3 p, float r);
    HD float get_radius();
    HD bool hits_ray(const Ray &r);

   private:
    Vector3 position;
    float radius;
};

#endif
