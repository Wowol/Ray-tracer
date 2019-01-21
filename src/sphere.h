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
        Vector3 oc = ray.get_position() - position;
        float a = ray.get_direction().scalar_product(ray.get_direction());
        float b = 2.0f * oc.scalar_product(ray.get_direction());
        float c = oc.scalar_product(oc) - radius * radius;

        float delta = b*b - 4*a*c;

        return delta > 0; 
    }

   private:
    Vector3 position;
    float radius;
};

#endif
