#if !defined(VECTOR3_HEADER)
#define VECTOR3_HEADER

#include <math.h>
#include <iostream>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Vector3 {
 public:
  float x, y, z;

  HD Vector3() : x(0), y(0), z(0) {}
  HD Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
  HD Vector3(const Vector3& other) : x(other.x), y(other.y), z(other.z) {}
  HD Vector3(const Vector3& begin, const Vector3& end)
      : x(end.x - begin.x), y(end.y - begin.y), z(end.z - begin.z) {}

  HD float scalar_product(const Vector3& second) const {
    return x * second.x + y * second.y + z * second.z;
  }

  HD Vector3 cross_product(const Vector3& second) const {
    return Vector3(y*second.z-z*second.y, z*second.x-x*second.z, x*second.y-y*second.x);
  }

  HD float distance(const Vector3& other) {
    return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2) +
                pow(z - other.z, 2));
  }
  HD float length() { return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)); }
  HD void normalize() {
    float multiplier = 1.0f / length();  // length should not be 0 :C
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;
  }

  HD Vector3 operator+(const Vector3& second) const {
    return Vector3(this->x + second.x, this->y + second.y, this->z + second.z);
  }

  HD Vector3 operator-(const Vector3& second) const {
    return Vector3(this->x - second.x, this->y - second.y, this->z - second.z);
  }

  HD Vector3 operator*(float scalar) const {
    return Vector3(this->x * scalar, this->y * scalar, this->z * scalar);
  }
};

#endif  // VECTOR3_HEADER