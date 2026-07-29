#include "mtmd-image.h"

#include <cstdio>
#include <vector>

static bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}

int main() {
    {
        const std::vector<int32_t> positions = mtmd_lladao_position_ids(70, 70);
        if (!check(positions.size() == 4900, "unexpected 70x70 position count")) {
            return 1;
        }
        for (int i = 0; i < 4900; i++) {
            if (!check(positions[i] == i, "70x70 position IDs are not row-major")) {
                return 1;
            }
        }
    }

    {
        const int                  rows      = 27;
        const int                  cols      = 35;
        const std::vector<int32_t> positions = mtmd_lladao_position_ids(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                const int expected = (70 * row / rows) * 70 + 70 * col / cols;
                if (!check(positions[row * cols + col] == expected, "dynamic position bucket mismatch")) {
                    return 1;
                }
            }
        }
    }

    struct image_size_case {
        clip_image_size input;
        clip_image_size expected;
    };

    const image_size_case cases[] = {
        { { 980, 980 },   { 980, 980 } },
        { { 800, 400 },   { 798, 406 } },
        { { 1000, 100 },  { 980, 98 }  },
        { { 2000, 1000 }, { 980, 490 } },
        { { 100, 100 },   { 378, 378 } },
        { { 1920, 1080 }, { 980, 546 } },
        { { 1080, 1920 }, { 546, 980 } },
        { { 1, 50 },      { 28, 980 }  },
    };

    for (const image_size_case & item : cases) {
        const clip_image_size actual = mtmd_lladao_image_size(item.input);
        if (!check(actual.width == item.expected.width && actual.height == item.expected.height,
                   "dynamic image size mismatch")) {
            return 1;
        }
        if (!check(actual.width % 14 == 0 && actual.height % 14 == 0, "dynamic image size is not patch aligned")) {
            return 1;
        }
        if (!check(actual.width <= 980 && actual.height <= 980, "dynamic image size exceeds 980")) {
            return 1;
        }
    }

    return 0;
}
