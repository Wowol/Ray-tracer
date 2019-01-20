
#if !defined(CAMERA_HEADER)
#define CAMERA_HEADER

#include "vector3.h"

class Camera {
   public:
    Camera(Vector3 pos, float distance_from_screen);
    Vector3 position;
    float distance_from_screen;  // represents field of view
};

#endif  // CAMERA_HEADER