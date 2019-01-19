#if !defined(VECTOR3_HEADER)
#define VECTOR3_HEADER

#include <iostream>

class Vector3 {
   public:
    float x, y, z;

    Vector3(float x, float y, float z);
    Vector3(const Vector3& other);
    Vector3(const Vector3& begin, const Vector3& end);

    float scalar_product(const Vector3& second) const;
    float distance(const Vector3& other);
    float length();
    void normalize();

    friend Vector3 operator+(const Vector3& first, const Vector3& other);
    friend Vector3 operator-(const Vector3& first, const Vector3& other);
    friend Vector3 operator*(const Vector3& first, float scalar);

    friend std::ostream& operator<<(std::ostream& stream, const Vector3& v);
};

#endif  // VECTOR3_HEADER