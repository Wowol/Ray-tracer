#ifndef SPHERE_HEADER
#define SPHERE_HEADER

#include "material.h"
#include "ray.h"
#include "vector3.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Sphere {
   public:
    HD Sphere(Vector3 pos, float r, Material const &material)
        : position(pos), radius(r), material(material) {}
    HD float get_radius() const { return radius; }
    HD Vector3 get_position() const { return position; }
    HD Material get_material() const { return material; }

    HD bool hits_ray(const Ray &ray) {
        Vector3 oc(ray.get_position(), position);
        float a = ray.get_direction().scalar_product(ray.get_direction());
        float b = 2.0f * oc.scalar_product(ray.get_direction());
        float c = oc.scalar_product(oc) - radius * radius;
        float delta = b * b - 4 * a * c;
        return delta >= 0;
    }

    HD Ray reflect(Ray const &ray) {
        Vector3 intersection_point = get_intersection_point(ray);
        Vector3 perpendicular(get_position(), intersection_point);
        perpendicular.normalize();

        Vector3 oc(ray.get_position(), get_position());

        Vector3 offset(ray.get_position(),
                       perpendicular * (perpendicular.scalar_product(oc)));

        return Ray(intersection_point, ray.get_position() + offset * 2);
    }

    HD Vector3 get_intersection_point(Ray const &ray) const {
        Vector3 oc(ray.get_position(), position);
        float center_distance = oc.cross_product(ray.get_direction()).length();
        float position_distance = oc.scalar_product(ray.get_direction());

        float offset =
            sqrtf(radius * radius - center_distance * center_distance);

        return ray.get_position() +
               ray.get_direction() * (position_distance - offset);
    }

   private:
    Vector3 position;
    float radius;
    Material material;
};

#endif
