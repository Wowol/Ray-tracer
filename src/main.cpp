#include "color.h"
#include "drawable.h"
#include "image.h"
#include "render.h"
#include <vector>

int main() {
    render(std::vector<Drawable *>()).writePNG("out.png");
    return 0;
}
