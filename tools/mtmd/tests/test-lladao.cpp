#include "mtmd-image.h"
#include "mtmd.h"

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

static bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}

int main() {
    {
        const mtmd_context_params params = mtmd_context_params_default();
        if (!check(!params.lladao_exact_tile, "exact tile preprocessing is not disabled by default")) {
            return 1;
        }
    }

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
        const std::pair<int, int> grids[] = {
            { 27, 35 },
            { 39, 70 },
            { 70, 39 },
        };
        for (const auto & [rows, cols] : grids) {
            const std::vector<int32_t> positions = mtmd_lladao_position_ids(rows, cols);
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    const int expected = row * 70 + col;
                    if (!check(positions[row * cols + col] == expected, "dynamic position ID mismatch")) {
                        return 1;
                    }
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

    {
        const image_size_case exact_cases[] = {
            { { 980, 980 }, { 980, 980 } },
            { { 979, 979 }, { 980, 980 } },
            { { 15, 20 },   { 28, 28 }   },
            { { 14, 14 },   { 14, 14 }   },
            { { 1, 1 },     { 14, 14 }   },
        };
        for (const image_size_case & item : exact_cases) {
            const clip_image_size actual = mtmd_lladao_exact_tile_size(item.input);
            if (!check(actual.width == item.expected.width && actual.height == item.expected.height,
                       "exact tile size mismatch")) {
                return 1;
            }
        }
    }

    {
        clip_image_u8 input;
        input.set_size({ 2, 1 }, false);
        input.set_pixel(0, 0, { 0, 128, 255 });
        input.set_pixel(1, 0, { 255, 0, 128 });

        const float mean[3] = { 0.5f, 0.5f, 0.5f };
        const float std[3]  = { 0.5f, 0.5f, 0.5f };
        const clip_image_f32 output = mtmd_lladao_exact_tile_image(input, mean, std);
        const std::vector<float> & pixels = output.get_ro_buf();

        if (!check(output.nx() == 14 && output.ny() == 14, "exact tile image is not patch aligned")) {
            return 1;
        }
        if (!check(pixels[0] == -1.0f && std::fabs(pixels[1] - 1.0f / 255.0f) < 1e-6f && pixels[2] == 1.0f,
                   "exact tile source pixel was not normalized")) {
            return 1;
        }
        if (!check(pixels[3] == 1.0f && pixels[4] == -1.0f && std::fabs(pixels[5] - 1.0f / 255.0f) < 1e-6f,
                   "exact tile source pixels were changed")) {
            return 1;
        }
        for (size_t i = 6; i < pixels.size(); i++) {
            if (!check(pixels[i] == 0.0f, "exact tile padding is not zero in normalized space")) {
                return 1;
            }
        }
    }

    return 0;
}
