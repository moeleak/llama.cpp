#include "../src/llama-d2f-mask.h"

#include <cstdlib>
#include <cstdint>
#include <vector>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static void assert_mask(
        const std::vector<int32_t> & positions,
        int32_t image_prefix_length,
        int32_t prefix_length,
        int32_t block_size,
        const std::vector<std::vector<bool>> & expected) {
    require(positions.size() == expected.size());

    for (int32_t query = 0; query < (int32_t) positions.size(); ++query) {
        require(expected[query].size() == positions.size());

        for (int32_t key = 0; key < (int32_t) positions.size(); ++key) {
            const bool visible = llm_d2f_attention_visible(
                    query, positions[query], key, positions[key],
                    image_prefix_length, prefix_length, block_size);
            require(visible == expected[query][key]);

            const bool position_visible = llm_d2f_attention_visible_position(
                    positions[query],
                    positions[key],
                    positions[image_prefix_length],
                    positions[prefix_length],
                    block_size);
            require(position_visible == visible);
        }
    }
}

static void test_cached_generation_slice() {
    const std::vector<int32_t> keys = { 0, 0, 2, 3, 4, 5, 6, 7 };
    const int32_t prompt_position = 2;
    const int32_t generation_position = 4;
    const int32_t block_size = 2;

    for (int32_t query_position = generation_position;
         query_position < (int32_t) keys.size();
         ++query_position) {
        for (int32_t key_position : keys) {
            const bool expected = key_position < generation_position ||
                    (key_position - generation_position) / block_size <=
                    (query_position - generation_position) / block_size;
            require(llm_d2f_attention_visible_position(
                        query_position,
                        key_position,
                        prompt_position,
                        generation_position,
                        block_size) == expected);
        }
    }
}

static void test_single_image() {
    assert_mask(
            { 0, 0, 0, 1, 2, 3, 4, 5 },
            3,
            5,
            2,
            {
                { true,  true,  true,  false, false, false, false, false },
                { true,  true,  true,  false, false, false, false, false },
                { true,  true,  true,  false, false, false, false, false },
                { true,  true,  true,  true,  true,  false, false, false },
                { true,  true,  true,  true,  true,  false, false, false },
                { true,  true,  true,  true,  true,  true,  true,  false },
                { true,  true,  true,  true,  true,  true,  true,  false },
                { true,  true,  true,  true,  true,  true,  true,  true  },
            });
}

static void test_multiple_images() {
    assert_mask(
            { 0, 0, 1, 1, 2, 3, 4, 5, 6, 7 },
            4,
            6,
            2,
            {
                { true,  true,  false, false, false, false, false, false, false, false },
                { true,  true,  false, false, false, false, false, false, false, false },
                { false, false, true,  true,  false, false, false, false, false, false },
                { false, false, true,  true,  false, false, false, false, false, false },
                { true,  true,  true,  true,  true,  true,  false, false, false, false },
                { true,  true,  true,  true,  true,  true,  false, false, false, false },
                { true,  true,  true,  true,  true,  true,  true,  true,  false, false },
                { true,  true,  true,  true,  true,  true,  true,  true,  false, false },
                { true,  true,  true,  true,  true,  true,  true,  true,  true,  true  },
                { true,  true,  true,  true,  true,  true,  true,  true,  true,  true  },
            });
}

int main() {
    test_single_image();
    test_multiple_images();
    test_cached_generation_slice();
    return 0;
}
