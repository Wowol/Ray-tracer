#ifndef SPHERE_HEADER
#define SPHERE_HEADER

#include "ray.h"
#include "vector3.h"

class Sphere {
   public:
    Sphere(Vector3 p, float r);
    float get_radius();
    bool hits_ray(const Ray &r);

   private:
    Vector3 position;
    float radius;
};

#endif
