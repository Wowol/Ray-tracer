#ifndef PW_RGBCOLOR_HEADER
#define PW_RGBCOLOR_HEADER

#include <cmath>

class RGBColor {
public:
    float data[4];

    inline float& r() { return data[0]; }
    inline float& g() { return data[1]; }
    inline float& b() { return data[2]; }

    inline float r() const { return data[0]; }
    inline float g() const { return data[1]; }
    inline float b() const { return data[2]; }

    RGBColor() {}
    RGBColor(float r, float g, float b) : data{r,g,b,0.0f} {}

    static RGBColor rep(float v) { return RGBColor(v,v,v); }

    template <typename Func>
    inline RGBColor transform(Func op) const {
      return RGBColor(op(r()), op(g()), op(b()));
    }

    RGBColor operator+(const RGBColor& c) const { return RGBColor(r()+c.r(), g()+c.g(), b()+c.b()); }
    RGBColor operator-(const RGBColor& c) const { return RGBColor(r()-c.r(), g()-c.g(), b()-c.b()); }
    RGBColor operator*(const RGBColor& c) const { return RGBColor(r()*c.r(), g()*c.g(), b()*c.b()); }
    
    bool operator==(const RGBColor& c) const { return r() == c.r() && g() == c.g() && b() == c.b(); }
    bool operator!=(const RGBColor& c) const { return !(*this == c); }

};

inline RGBColor operator*(float scalar, const RGBColor& b) { return RGBColor::rep(scalar)*b; }
inline RGBColor operator*(const RGBColor& a, float scalar) { return a*RGBColor::rep(scalar); }
inline RGBColor operator/(const RGBColor& a, float scalar) { return a*RGBColor::rep(1.0f/scalar); }

inline RGBColor linear2RGB(const RGBColor& color) {
    return color.transform([](float v) {
        if (v>=0.0f)
            return std::pow(v, 1/2.2f);
        else
            return -std::pow(-v, 1/2.2f);
    });
} 
inline RGBColor RGB2Linear(const RGBColor& color) {
    return color.transform([](float v) {
        if (v>=0.0f)
            return std::pow(v, 2.2f);
        else
            return -std::pow(-v, 2.2f);
    });
}

inline RGBColor clamp(const RGBColor& color) {
    return color.transform([](float v) {
        if (v > 1.0f)
            return 1.0f;
        if (v < 0.0f)
            return 0.0f;
        return v;
    });
};


#endif
