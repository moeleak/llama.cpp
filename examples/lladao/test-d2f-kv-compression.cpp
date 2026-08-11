#include "lladao-d2f-kv-compression.h"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace lladao::detail;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static bool close(float left, float right, float tolerance = 1e-6f) {
    return std::abs(left - right) <= tolerance;
}

int main() {
    const d2f_visual_kv_compression_config defaults;
    require(
            defaults.vision_tile_size == 16 && defaults.vision_topk_tiles == 20 &&
            close(defaults.vision_token_keep_ratio, 0.75f) &&
            defaults.vision_score_query_window == 32 && defaults.vision_score_layers == 4 &&
            defaults.vision_score_layer_mode == d2f_kv_score_layer_mode::last &&
            defaults.vision_score_pool_kernel == 7,
            "visual KV compression defaults changed");

    const auto tiles = build_vision_tiles(3, 5, 2);
    require(tiles.size() == 6, "vision tile count is invalid");
    require(
            tiles[0] == std::vector<int32_t>({ 0, 1, 5, 6 }) &&
            tiles[1] == std::vector<int32_t>({ 2, 3, 7, 8 }) &&
            tiles[2] == std::vector<int32_t>({ 4, 9 }) &&
            tiles[3] == std::vector<int32_t>({ 10, 11 }),
            "vision tile row-major layout is invalid");

    const std::vector<float> pooled = max_pool_vision_scores_2d(
            { -9.0f, -8.0f, -7.0f, -6.0f, 5.0f, -4.0f, -3.0f, -2.0f, -1.0f },
            3,
            3,
            3);
    require(
            close(pooled[0], 5.0f) && close(pooled[4], 5.0f) && close(pooled[8], 5.0f),
            "vision max pooling does not match padded 2D max pooling");

    const auto selected_tiles = select_top_vision_tiles(
            { 0.0f, 1.0f, 4.0f, 3.0f },
            build_vision_tiles(2, 2, 1),
            2);
    require(
            selected_tiles == std::vector<int32_t>({ 2, 3 }),
            "vision tile Top-K selection is invalid");

    require(
            select_visual_score_layers(8, 4, d2f_kv_score_layer_mode::last) ==
                    std::vector<int32_t>({ 4, 5, 6, 7 }) &&
            select_visual_score_layers(3, 2, d2f_kv_score_layer_mode::first) ==
                    std::vector<int32_t>({ 0, 1 }),
            "visual score layer selection is invalid");

    const d2f_visual_prefix_layout layout = {
        {
            { 0, 5, 1, 4, 1, 3, 0 },
            { 5, 11, 6, 10, 2, 2, 5 },
        },
        11,
        13,
        11,
    };
    d2f_visual_layer_scores layer_one;
    layer_one.layer_index = 1;
    layer_one.num_kv_heads = 2;
    layer_one.values = {
        0.0f, 0.0f, 10.0f, 9.0f, 8.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 10.0f, 0.0f, 1.0f, 8.0f, 9.0f,
    };
    d2f_visual_layer_scores layer_three;
    layer_three.layer_index = 3;
    layer_three.num_kv_heads = 2;
    layer_three.values = {
        0.0f, 0.0f, 10.0f, 0.0f, 1.0f, 8.0f, 9.0f,
        0.0f, 0.0f, 10.0f, 9.0f, 8.0f, 1.0f, 0.0f,
    };
    d2f_visual_kv_compression_config config;
    config.vision_tile_size = 2;
    config.vision_topk_tiles = 1;
    config.vision_token_keep_ratio = 0.5f;
    config.vision_score_pool_kernel = 1;

    const d2f_visual_kv_keep_plan plan = build_visual_kv_keep_plan(
            4,
            layout,
            { layer_one, layer_three },
            config);
    require(
            plan.num_layers == 4 && plan.num_kv_heads == 2 && plan.keep_count == 9 &&
            plan.destination_positions == std::vector<int32_t>({ 0, 0, 0, 5, 5, 5, 5, 11, 12 }),
            "visual KV keep-plan shape is invalid");
    require(
            plan.stats.dense_prefix_tokens == 13 && plan.stats.cached_prefix_tokens == 9 &&
            plan.stats.vision_patches == 7 && plan.stats.vision_kept_patches == 3 &&
            plan.stats.vision_tiles == 3 && plan.stats.vision_selected_tiles == 2 &&
            plan.stats.candidate_patches == 5 && close(plan.stats.compression_ratio, 9.0f / 13.0f),
            "visual KV compression statistics are invalid");
    const std::vector<int32_t> layer_one_head_zero = { 0, 3, 4, 5, 6, 7, 10, 11, 12 };
    const std::vector<int32_t> layer_one_head_one = { 0, 3, 4, 5, 8, 9, 10, 11, 12 };
    for (int32_t destination = 0; destination < plan.keep_count; ++destination) {
        require(
                plan.source_index(0, 0, destination) == layer_one_head_zero[destination] &&
                plan.source_index(2, 0, destination) == layer_one_head_zero[destination] &&
                plan.source_index(0, 1, destination) == layer_one_head_one[destination] &&
                plan.source_index(3, 0, destination) == layer_one_head_one[destination],
                "per-layer or per-head visual patch selection is invalid");
    }

    const d2f_visual_prefix_layout attention_layout = {
        { { 0, 5, 1, 4, 1, 3, 0 } },
        5,
        6,
        5,
    };
    const std::vector<float> query = { 1.0f, 1.0f };
    const std::vector<float> key = { 0.0f, 3.0f, 2.0f, 1.0f, 0.0f, 0.0f };
    const d2f_visual_layer_scores attention_scores = compute_visual_attention_scores(
            {
                2,
                1,
                6,
                2,
                1,
                1,
                1.0f,
                query.data(),
                key.data(),
            },
            attention_layout,
            32);
    const double denominator = 3.0 + std::exp(1.0) + std::exp(2.0) + std::exp(3.0);
    require(
            attention_scores.values.size() == 3 &&
            close(attention_scores.values[0], static_cast<float>(std::exp(3.0) / denominator)) &&
            close(attention_scores.values[1], static_cast<float>(std::exp(2.0) / denominator)) &&
            close(attention_scores.values[2], static_cast<float>(std::exp(1.0) / denominator)),
            "post-RoPE Q/K visual attention scoring is invalid");

    return 0;
}
