
#if !defined(RAY_HEADER)
#define RAY_HEADER

#include "vector3.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Ray {
   public:
    HD Ray(Vector3 p, Vector3 d) : position(p), direction(d) { direction.normalize(); }
    HD float distance_to_point(const Vector3& point) const {
        Vector3 vector_between = Vector3(position, point);
        Vector3 puv = direction * direction.scalar_product(vector_between);
        return (point - puv).length();
    }

   private:
    Vector3 position;
    Vector3 direction;
};

#endif  // RAY_HEADER
