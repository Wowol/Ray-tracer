#ifndef PW_RGBCOLOR_HEADER
#define PW_RGBCOLOR_HEADER

#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class RGBColor {
  public:
    float data[4];

    HD inline float &r() { return data[0]; }
    HD inline float &g() { return data[1]; }
    HD inline float &b() { return data[2]; }

    HD inline float r() const { return data[0]; }
    HD inline float g() const { return data[1]; }
    HD inline float b() const { return data[2]; }

    HD RGBColor() {}
    HD RGBColor(float r, float g, float b) : data{r, g, b, 0.0f} {}

    HD static RGBColor rep(float v) { return RGBColor(v, v, v); }

    template <typename Func> HD inline RGBColor transform(Func op) const {
        return RGBColor(op(r()), op(g()), op(b()));
    }

    HD RGBColor operator+(const RGBColor &c) const {
        return RGBColor(r() + c.r(), g() + c.g(), b() + c.b());
    }
    HD RGBColor operator-(const RGBColor &c) const {
        return RGBColor(r() - c.r(), g() - c.g(), b() - c.b());
    }
    HD RGBColor operator*(const RGBColor &c) const {
        return RGBColor(r() * c.r(), g() * c.g(), b() * c.b());
    }

    HD bool operator==(const RGBColor &c) const {
        return r() == c.r() && g() == c.g() && b() == c.b();
    }
    HD bool operator!=(const RGBColor &c) const { return !(*this == c); }
};

HD inline RGBColor operator*(float scalar, const RGBColor &b) {
    return RGBColor::rep(scalar) * b;
}
HD inline RGBColor operator*(const RGBColor &a, float scalar) {
    return a * RGBColor::rep(scalar);
}
HD inline RGBColor operator/(const RGBColor &a, float scalar) {
    return a * RGBColor::rep(1.0f / scalar);
}

HD inline RGBColor linear2RGB(const RGBColor &color) {
    return color.transform([](float v) {
        if (v >= 0.0f)
            return std::pow(v, 1 / 2.2f);
        else
            return -std::pow(-v, 1 / 2.2f);
    });
}
HD inline RGBColor RGB2Linear(const RGBColor &color) {
    return color.transform([](float v) {
        if (v >= 0.0f)
            return std::pow(v, 2.2f);
        else
            return -std::pow(-v, 2.2f);
    });
}

HD inline RGBColor clamp(const RGBColor &color) {
    return color.transform([](float v) {
        if (v > 1.0f)
            return 1.0f;
        if (v < 0.0f)
            return 0.0f;
        return v;
    });
};

#endif
