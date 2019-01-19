
#if !defined(RAY_HEADER)
#define RAY_HEADER

#include "vector3.h"

class Ray {
   public:
    Ray(Vector3 position, Vector3 direction);
    float distance_to_point(const Vector3& point) const;

   private:
    Vector3 position;
    Vector3 direction;
};

#endif  // RAY_HEADER
