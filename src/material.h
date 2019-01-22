#include "color.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Material {
  public:
    Material(RGBColor const &color, float const &reflection_coefficient,
             float const &transparency_coefficient)
        : color(color), reflection_coefficient(reflection_coefficient),
          transparency_coefficient(transparency_coefficient){};

    HD float get_reflection_coefficient() { return reflection_coefficient; }
    HD float get_transparency_coefficient() { return transparency_coefficient; }
    HD RGBColor get_color() { return color; }

  private:
    RGBColor color;
    float reflection_coefficient;
    float transparency_coefficient;
};
