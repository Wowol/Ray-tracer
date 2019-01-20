#if !defined(VECTOR3_HEADER)
#define VECTOR3_HEADER

#include <iostream>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Vector3 {
   public:
    float x, y, z;

    HD Vector3();
    HD Vector3(float x, float y, float z);
    HD Vector3(const Vector3& other);
    HD Vector3(const Vector3& begin, const Vector3& end);

    HD float scalar_product(const Vector3& second) const;
    HD float distance(const Vector3& other);
    HD float length();
    HD void normalize();

    HD friend Vector3 operator+(const Vector3& first, const Vector3& other);
    HD friend Vector3 operator-(const Vector3& first, const Vector3& other);
    HD friend Vector3 operator*(const Vector3& first, float scalar);

    HD friend std::ostream& operator<<(std::ostream& stream, const Vector3& v);
};

#endif  // VECTOR3_HEADER