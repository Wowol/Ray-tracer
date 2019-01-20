
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
    HD Ray(Vector3 position, Vector3 direction);
    HD float distance_to_point(const Vector3& point) const;

   private:
    Vector3 position;
    Vector3 direction;
};

#endif  // RAY_HEADER
