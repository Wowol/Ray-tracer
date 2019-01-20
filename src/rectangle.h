
#if !defined(RECTANGLE_HEADER)
#define RECTANGLE_HEADER

#include "vector3.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Rectangle {
   public:
    HD Rectangle(Vector3 left_top, Vector3 right_bottom);
    HD Rectangle(Vector3 center, float width, float height);
    Vector3 left_top_point;
    Vector3 right_bottom_point;
    HD float width() const;
    HD float height() const;

    friend std::ostream& operator<<(std::ostream& stream, const Rectangle& r);
};

#endif  // RECTANGLE_HEADER
