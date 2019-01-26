
#if !defined(LIGHT_HEADER)
#define LIGHT_HEADER

#include "vector3.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Light {
   public:
    Vector3 position;
    Vector3 intensity;
    HD Light(const Vector3 &p, const Vector3 &i) : position(p), intensity(i) {}
};

#endif  // LIGHT_HEADER
