
#include "rectangle.h"
#include "vector3.h"

Vector3 left_top_point;
Vector3 right_bottom_point;

Rectangle::Rectangle(Vector3 center, float width, float height) {
    left_top_point = Vector3(center.x - width / 2, center.y + height / 2, center.z);
    right_bottom_point = Vector3(center.x + width / 2, center.y - height / 2, center.z);
}
Rectangle::Rectangle(Vector3 left_top, Vector3 right_bottom)
    : left_top_point(left_top), right_bottom_point(right_bottom) {}

float Rectangle::width() const { return right_bottom_point.x - left_top_point.x; }

float Rectangle::height() const { return left_top_point.y - right_bottom_point.y; }

std::ostream &operator<<(std::ostream &stream, const Rectangle &r) {
    stream << "Width: " << r.width() << std::endl;
    stream << "Height: " << r.height() << std::endl;
    stream << "Left top: " << r.left_top_point << std::endl;
    stream << "Right bottom: " << r.right_bottom_point << std::endl;
    return stream;
}
