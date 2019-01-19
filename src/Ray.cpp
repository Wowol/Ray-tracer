#include "ray.h"
#include <math.h>
#include "sphere.h"
#include "vector3.h"

#include <iostream>

Ray::Ray(Vector3 p, Vector3 d) : position(p), direction(d) {
    direction.normalize();
}

float Ray::distance_to_point(const Vector3& point) const {
    Vector3 vector_between = Vector3(position, point);
    Vector3 puv = direction * direction.scalar_product(vector_between);
    return (point - puv).length();
}