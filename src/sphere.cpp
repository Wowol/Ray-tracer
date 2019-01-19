#include "sphere.h"
#include "ray.h"

Sphere::Sphere(Vector3 pos, float r) : position(pos), radius(r) {}

float Sphere::get_radius() {
    return radius;
}

bool Sphere::hits_ray(const Ray &ray) {
    return ray.distance_to_point(position) <= radius;
}