#include "vector3.h"
#include <math.h>
#include <iostream>

Vector3::Vector3() : x(0), y(0), z(0) {}
Vector3::Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
Vector3::Vector3(const Vector3 &other) : x(other.x), y(other.y), z(other.z) {}
Vector3::Vector3(const Vector3 &begin, const Vector3 &end) : x(end.x - begin.x), y(end.y - begin.y), z(end.z - begin.z) {}

float Vector3::distance(const Vector3 &other) {
    return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2) + pow(z - other.z, 2));
}

float Vector3::length() {
    return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
}

void Vector3::normalize() {
    float multiplier = 1.0f / length();  // length should not be 0 :C
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;
}

float Vector3::scalar_product(const Vector3 &second) const {
    return x * second.x + y * second.y + z * second.z;
}

Vector3 operator+(const Vector3 &first, const Vector3 &second) {
    return Vector3(first.x + second.x, first.y + second.y, first.z + second.z);
}

Vector3 operator-(const Vector3 &first, const Vector3 &second) {
    return Vector3(first.x - second.x, first.y - second.y, first.z - second.z);
}

Vector3 operator*(const Vector3 &first, float scalar) {
    return Vector3(first.x * scalar, first.y * scalar, first.z * scalar);
}

std::ostream &operator<<(std::ostream &stream, const Vector3 &v) {
    stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return stream;
}
