
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
    HD Camera(Vector3 pos, Vector3 vector_to_screen, float screen_width, float screen_height)
        : position(pos),
          vector_to_screen(vector_to_screen),
          screen_width(screen_width),
          screen_height(screen_height),
          screen(Rectangle(Vector3(position.x + vector_to_screen.x, position.y + vector_to_screen.y,
                                   position.z + vector_to_screen.z),
                           screen_width, screen_height)) {}
    Vector3 position;
    Vector3 vector_to_screen;
    float screen_width;
    float screen_height;
    HD Rectangle get_screen() { return screen; }

   private:
    Rectangle screen;
};

#endif  // CAMERA_HEADER