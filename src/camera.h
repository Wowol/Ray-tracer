
#if !defined(CAMERA_HEADER)
#define CAMERA_HEADER

#include "rectangle.h"
#include "vector3.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Camera {
   public:
    HD Camera(Vector3 pos, Vector3 vector_to_screen, float screen_width, float screen_height);
    Vector3 position;
    Vector3 vector_to_screen;
    float screen_width;
    float screen_height;
    HD Rectangle get_screen();
   
   private:
    Rectangle screen;
};

#endif  // CAMERA_HEADER