
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
    HD Rectangle(Vector3 left_top, Vector3 right_bottom) : left_top_point(left_top), right_bottom_point(right_bottom) {}
    HD Rectangle(Vector3 center, float width, float height) {
        left_top_point = Vector3(center.x - width / 2, center.y + height / 2, center.z);
        right_bottom_point = Vector3(center.x + width / 2, center.y - height / 2, center.z);
    }
    Vector3 left_top_point;
    Vector3 right_bottom_point;
    HD float width() const { return right_bottom_point.x - left_top_point.x; }
    HD float height() const { return left_top_point.y - right_bottom_point.y; }

};

#endif  // RECTANGLE_HEADER
