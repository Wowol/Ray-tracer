
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
    RGBColor emitting_light;
    HD Light(const Vector3 &p, const RGBColor &i) : position(p), emitting_light(i) {}
};

#endif  // LIGHT_HEADER
