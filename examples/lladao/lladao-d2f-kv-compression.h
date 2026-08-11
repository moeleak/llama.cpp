#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lladao::detail {

enum class d2f_kv_score_layer_mode : int32_t {
    all,
    first,
    last,
};

struct d2f_visual_kv_compression_config {
    int32_t vision_tile_size = 16;
    int32_t vision_topk_tiles = 20;
    float vision_token_keep_ratio = 0.75f;
    int32_t vision_score_query_window = 32;
    int32_t vision_score_layers = 4;
    d2f_kv_score_layer_mode vision_score_layer_mode = d2f_kv_score_layer_mode::last;
    int32_t vision_score_pool_kernel = 7;
};

struct d2f_visual_span {
    int32_t token_start = 0;
    int32_t token_end = 0;
    int32_t patch_start = 0;
    int32_t patch_end = 0;
    int32_t grid_height = 0;
    int32_t grid_width = 0;
    int32_t position = 0;
};

struct d2f_visual_prefix_layout {
    std::vector<d2f_visual_span> spans;
    int32_t prompt_start = 0;
    int32_t prefix_length = 0;
    int32_t prompt_position = 0;
};

// Scores are row-major [num_kv_heads][total_visual_patches].
struct d2f_visual_layer_scores {
    int32_t layer_index = -1;
    int32_t num_kv_heads = 0;
    std::vector<float> values;
};

// Query is row-major [query_tokens][num_query_heads][head_dim].
// Key is row-major [prefix_tokens][num_kv_heads][head_dim].
struct d2f_post_rope_qk_view {
    int32_t layer_index = -1;
    int32_t query_tokens = 0;
    int32_t prefix_tokens = 0;
    int32_t num_query_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    float attention_scale = 1.0f;
    const float * query = nullptr;
    const float * key = nullptr;
};

struct d2f_visual_kv_compression_stats {
    int32_t dense_prefix_tokens = 0;
    int32_t cached_prefix_tokens = 0;
    int32_t vision_patches = 0;
    int32_t vision_kept_patches = 0;
    int32_t vision_tiles = 0;
    int32_t vision_selected_tiles = 0;
    int32_t candidate_patches = 0;
    float compression_ratio = 1.0f;
};

struct d2f_visual_kv_keep_plan {
    int32_t num_layers = 0;
    int32_t num_kv_heads = 0;
    int32_t keep_count = 0;

    // Row-major [num_layers][num_kv_heads][keep_count]. Each value is a
    // source token index in the dense prefix.
    std::vector<int32_t> source_indices;

    // All layers and heads write their j-th source to destination_positions[j].
    std::vector<int32_t> destination_positions;
    std::vector<std::vector<int32_t>> selected_tiles_by_span;
    d2f_visual_kv_compression_stats stats;

    int32_t source_index(int32_t layer, int32_t kv_head, int32_t destination) const {
        if (layer < 0 || layer >= num_layers || kv_head < 0 || kv_head >= num_kv_heads ||
            destination < 0 || destination >= keep_count) {
            throw std::out_of_range("visual KV keep-plan index is out of range");
        }
        const size_t offset =
                (static_cast<size_t>(layer) * num_kv_heads + kv_head) * keep_count + destination;
        return source_indices.at(offset);
    }
};

inline void validate_visual_kv_compression_config(const d2f_visual_kv_compression_config & config) {
    if (config.vision_tile_size <= 0 || config.vision_topk_tiles < 0 ||
        !(config.vision_token_keep_ratio > 0.0f && config.vision_token_keep_ratio <= 1.0f) ||
        config.vision_score_query_window < 0 || config.vision_score_layers < 0 ||
        config.vision_score_pool_kernel <= 0 || config.vision_score_pool_kernel % 2 == 0) {
        throw std::invalid_argument("invalid visual KV compression configuration");
    }
}

inline int32_t validate_visual_prefix_layout(const d2f_visual_prefix_layout & layout) {
    if (layout.spans.empty() || layout.prompt_start < 0 || layout.prefix_length <= layout.prompt_start) {
        throw std::invalid_argument("invalid visual prefix layout");
    }

    int64_t total_patches = 0;
    int32_t expected_token_start = 0;
    for (const d2f_visual_span & span : layout.spans) {
        const int64_t patch_count = static_cast<int64_t>(span.grid_height) * span.grid_width;
        if (span.grid_height <= 0 || span.grid_width <= 0 || patch_count > std::numeric_limits<int32_t>::max() ||
            span.token_start != expected_token_start || span.patch_start != span.token_start + 1 ||
            span.patch_end != span.token_end - 1 || span.patch_end - span.patch_start != patch_count) {
            throw std::invalid_argument("invalid visual span layout");
        }
        expected_token_start = span.token_end;
        total_patches += patch_count;
    }
    if (expected_token_start != layout.prompt_start || total_patches > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("visual spans do not end at the prompt boundary");
    }
    return static_cast<int32_t>(total_patches);
}

inline std::vector<int32_t> select_visual_score_layers(
        int32_t total_layers,
        int32_t layer_window,
        d2f_kv_score_layer_mode layer_mode) {
    if (total_layers <= 0 || layer_window < 0) {
        throw std::invalid_argument("invalid visual score layer selection");
    }
    if (layer_mode == d2f_kv_score_layer_mode::all || layer_window == 0) {
        std::vector<int32_t> result(total_layers);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    const int32_t window = std::min(total_layers, std::max(1, layer_window));
    int32_t first = 0;
    if (layer_mode == d2f_kv_score_layer_mode::last) {
        first = total_layers - window;
    } else if (layer_mode != d2f_kv_score_layer_mode::first) {
        throw std::invalid_argument("unsupported visual score layer mode");
    }

    std::vector<int32_t> result(window);
    std::iota(result.begin(), result.end(), first);
    return result;
}

inline std::vector<std::vector<int32_t>> build_vision_tiles(
        int32_t grid_height,
        int32_t grid_width,
        int32_t tile_size) {
    if (grid_height <= 0 || grid_width <= 0 || tile_size <= 0) {
        throw std::invalid_argument("invalid vision tile dimensions");
    }

    const int32_t tile_rows = (grid_height + tile_size - 1) / tile_size;
    const int32_t tile_columns = (grid_width + tile_size - 1) / tile_size;
    std::vector<std::vector<int32_t>> tiles(static_cast<size_t>(tile_rows) * tile_columns);
    for (int32_t row = 0; row < grid_height; ++row) {
        for (int32_t column = 0; column < grid_width; ++column) {
            const int32_t tile_row = row / tile_size;
            const int32_t tile_column = column / tile_size;
            tiles[static_cast<size_t>(tile_row) * tile_columns + tile_column].push_back(
                    row * grid_width + column);
        }
    }
    return tiles;
}

inline std::vector<float> max_pool_vision_scores_2d(
        const std::vector<float> & scores,
        int32_t grid_height,
        int32_t grid_width,
        int32_t kernel_size) {
    const int64_t patch_count = static_cast<int64_t>(grid_height) * grid_width;
    if (grid_height <= 0 || grid_width <= 0 || kernel_size <= 0 || kernel_size % 2 == 0 ||
        patch_count != static_cast<int64_t>(scores.size())) {
        throw std::invalid_argument("invalid vision score pooling input");
    }
    for (float value : scores) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("vision scores must be finite");
        }
    }

    const int32_t radius = kernel_size / 2;
    std::vector<float> result(scores.size(), -std::numeric_limits<float>::infinity());
    for (int32_t row = 0; row < grid_height; ++row) {
        for (int32_t column = 0; column < grid_width; ++column) {
            float maximum = -std::numeric_limits<float>::infinity();
            const int32_t first_row = std::max(0, row - radius);
            const int32_t last_row = std::min(grid_height - 1, row + radius);
            const int32_t first_column = std::max(0, column - radius);
            const int32_t last_column = std::min(grid_width - 1, column + radius);
            for (int32_t source_row = first_row; source_row <= last_row; ++source_row) {
                for (int32_t source_column = first_column; source_column <= last_column; ++source_column) {
                    maximum = std::max(maximum, scores[static_cast<size_t>(source_row) * grid_width + source_column]);
                }
            }
            result[static_cast<size_t>(row) * grid_width + column] = maximum;
        }
    }
    return result;
}

inline std::vector<int32_t> select_top_vision_tiles(
        const std::vector<float> & patch_scores,
        const std::vector<std::vector<int32_t>> & tiles,
        int32_t topk_tiles) {
    if (topk_tiles < 0) {
        throw std::invalid_argument("vision tile Top-K must be non-negative");
    }
    if (tiles.empty()) {
        return {};
    }
    for (float score : patch_scores) {
        if (!std::isfinite(score)) {
            throw std::invalid_argument("vision tile scores must be finite");
        }
    }
    for (const std::vector<int32_t> & tile : tiles) {
        if (tile.empty()) {
            throw std::invalid_argument("vision tiles must not be empty");
        }
        for (int32_t patch : tile) {
            if (patch < 0 || patch >= static_cast<int32_t>(patch_scores.size())) {
                throw std::invalid_argument("vision tile patch index is out of range");
            }
        }
    }

    if (topk_tiles == 0 || topk_tiles >= static_cast<int32_t>(tiles.size())) {
        std::vector<int32_t> result(tiles.size());
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    std::vector<std::pair<float, int32_t>> ranked;
    ranked.reserve(tiles.size());
    for (int32_t tile_index = 0; tile_index < static_cast<int32_t>(tiles.size()); ++tile_index) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int32_t patch : tiles[tile_index]) {
            maximum = std::max(maximum, patch_scores[patch]);
        }
        ranked.emplace_back(maximum, tile_index);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto & left, const auto & right) {
        return left.first != right.first ? left.first > right.first : left.second < right.second;
    });

    std::vector<int32_t> selected;
    selected.reserve(topk_tiles);
    for (int32_t index = 0; index < topk_tiles; ++index) {
        selected.push_back(ranked[index].second);
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

inline int32_t nearest_scored_layer(
        const std::vector<d2f_visual_layer_scores> & sorted_scores,
        int32_t model_layer) {
    if (sorted_scores.empty()) {
        throw std::invalid_argument("visual KV compression needs at least one scored layer");
    }
    int32_t best = 0;
    int32_t best_distance = std::abs(sorted_scores[0].layer_index - model_layer);
    for (int32_t index = 1; index < static_cast<int32_t>(sorted_scores.size()); ++index) {
        const int32_t distance = std::abs(sorted_scores[index].layer_index - model_layer);
        if (distance < best_distance) {
            best = index;
            best_distance = distance;
        }
    }
    return best;
}

inline d2f_visual_layer_scores compute_visual_attention_scores(
        const d2f_post_rope_qk_view & view,
        const d2f_visual_prefix_layout & layout,
        int32_t query_window) {
    const int32_t total_patches = validate_visual_prefix_layout(layout);
    const int32_t prompt_tokens = layout.prefix_length - layout.prompt_start;
    if (view.layer_index < 0 || view.query == nullptr || view.key == nullptr ||
        view.query_tokens != prompt_tokens || view.prefix_tokens != layout.prefix_length ||
        view.num_query_heads <= 0 || view.num_kv_heads <= 0 || view.head_dim <= 0 ||
        view.num_query_heads % view.num_kv_heads != 0 || query_window < 0 ||
        !std::isfinite(view.attention_scale)) {
        throw std::invalid_argument("invalid post-RoPE Q/K score view");
    }

    int32_t query_end = std::max(1, view.query_tokens - 1);
    int32_t query_start = 1;
    if (query_end <= query_start) {
        query_start = 0;
        query_end = view.query_tokens;
    }
    if (query_window > 0) {
        query_start = std::max(query_start, query_end - query_window);
    }

    const int32_t query_group_size = view.num_query_heads / view.num_kv_heads;
    d2f_visual_layer_scores result;
    result.layer_index = view.layer_index;
    result.num_kv_heads = view.num_kv_heads;
    result.values.assign(static_cast<size_t>(view.num_kv_heads) * total_patches, 0.0f);

    std::vector<float> logits(view.prefix_tokens);
    for (int32_t query_index = query_start; query_index < query_end; ++query_index) {
        for (int32_t query_head = 0; query_head < view.num_query_heads; ++query_head) {
            const int32_t kv_head = query_head / query_group_size;
            const float * query = view.query +
                    (static_cast<size_t>(query_index) * view.num_query_heads + query_head) * view.head_dim;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int32_t token = 0; token < view.prefix_tokens; ++token) {
                const float * key = view.key +
                        (static_cast<size_t>(token) * view.num_kv_heads + kv_head) * view.head_dim;
                double dot = 0.0;
                for (int32_t dimension = 0; dimension < view.head_dim; ++dimension) {
                    dot += static_cast<double>(query[dimension]) * key[dimension];
                }
                const float logit = static_cast<float>(dot * view.attention_scale);
                if (!std::isfinite(logit)) {
                    throw std::invalid_argument("post-RoPE Q/K produced a non-finite attention logit");
                }
                logits[token] = logit;
                maximum = std::max(maximum, logit);
            }

            double denominator = 0.0;
            for (float logit : logits) {
                denominator += std::exp(static_cast<double>(logit - maximum));
            }
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                throw std::runtime_error("visual attention softmax normalization failed");
            }
            const double weight = 1.0 / (denominator * query_group_size);

            int32_t score_offset = 0;
            for (const d2f_visual_span & span : layout.spans) {
                for (int32_t patch = span.patch_start; patch < span.patch_end; ++patch) {
                    result.values[static_cast<size_t>(kv_head) * total_patches + score_offset] +=
                            static_cast<float>(std::exp(static_cast<double>(logits[patch] - maximum)) * weight);
                    ++score_offset;
                }
            }
        }
    }
    return result;
}

inline d2f_visual_kv_keep_plan build_visual_kv_keep_plan(
        int32_t num_model_layers,
        const d2f_visual_prefix_layout & layout,
        std::vector<d2f_visual_layer_scores> scores_by_layer,
        const d2f_visual_kv_compression_config & config = {}) {
    validate_visual_kv_compression_config(config);
    const int32_t total_patches = validate_visual_prefix_layout(layout);
    if (num_model_layers <= 0 || scores_by_layer.empty()) {
        throw std::invalid_argument("visual KV compression requires model layers and scores");
    }

    std::sort(scores_by_layer.begin(), scores_by_layer.end(), [](const auto & left, const auto & right) {
        return left.layer_index < right.layer_index;
    });
    const int32_t num_kv_heads = scores_by_layer.front().num_kv_heads;
    if (num_kv_heads <= 0) {
        throw std::invalid_argument("visual KV scores must contain KV heads");
    }
    int32_t previous_layer = -1;
    for (const d2f_visual_layer_scores & layer : scores_by_layer) {
        if (layer.layer_index < 0 || layer.layer_index >= num_model_layers ||
            layer.layer_index == previous_layer || layer.num_kv_heads != num_kv_heads ||
            layer.values.size() != static_cast<size_t>(num_kv_heads) * total_patches) {
            throw std::invalid_argument("inconsistent visual layer scores");
        }
        for (float value : layer.values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("visual layer scores must be finite");
            }
        }
        previous_layer = layer.layer_index;
    }

    std::vector<d2f_visual_layer_scores> pooled_scores = scores_by_layer;
    int32_t score_offset = 0;
    for (const d2f_visual_span & span : layout.spans) {
        const int32_t span_patches = span.patch_end - span.patch_start;
        for (d2f_visual_layer_scores & layer : pooled_scores) {
            for (int32_t head = 0; head < num_kv_heads; ++head) {
                const size_t begin = static_cast<size_t>(head) * total_patches + score_offset;
                std::vector<float> local(
                        layer.values.begin() + begin,
                        layer.values.begin() + begin + span_patches);
                const std::vector<float> pooled = max_pool_vision_scores_2d(
                        local,
                        span.grid_height,
                        span.grid_width,
                        config.vision_score_pool_kernel);
                std::copy(pooled.begin(), pooled.end(), layer.values.begin() + begin);
            }
        }
        score_offset += span_patches;
    }

    std::vector<float> aggregate_scores(total_patches, 0.0f);
    const float aggregate_scale = 1.0f /
            (static_cast<float>(pooled_scores.size()) * num_kv_heads);
    for (const d2f_visual_layer_scores & layer : pooled_scores) {
        for (int32_t head = 0; head < num_kv_heads; ++head) {
            for (int32_t patch = 0; patch < total_patches; ++patch) {
                aggregate_scores[patch] +=
                        layer.values[static_cast<size_t>(head) * total_patches + patch] * aggregate_scale;
            }
        }
    }

    struct span_plan {
        int32_t score_start = 0;
        int32_t patch_start = 0;
        int32_t keep_count = 0;
        std::vector<int32_t> candidates;
    };
    std::vector<span_plan> span_plans;
    d2f_visual_kv_keep_plan result;
    result.num_layers = num_model_layers;
    result.num_kv_heads = num_kv_heads;
    result.stats.dense_prefix_tokens = layout.prefix_length;
    result.stats.vision_patches = total_patches;

    score_offset = 0;
    for (const d2f_visual_span & span : layout.spans) {
        const int32_t span_patches = span.patch_end - span.patch_start;
        const std::vector<std::vector<int32_t>> tiles = build_vision_tiles(
                span.grid_height,
                span.grid_width,
                config.vision_tile_size);
        const std::vector<float> local_aggregate(
                aggregate_scores.begin() + score_offset,
                aggregate_scores.begin() + score_offset + span_patches);
        const std::vector<int32_t> selected_tiles = select_top_vision_tiles(
                local_aggregate,
                tiles,
                config.vision_topk_tiles);

        std::vector<int32_t> candidates;
        for (int32_t tile : selected_tiles) {
            candidates.insert(candidates.end(), tiles[tile].begin(), tiles[tile].end());
        }
        std::sort(candidates.begin(), candidates.end());
        if (candidates.empty()) {
            throw std::runtime_error("visual tile selection produced no patch candidates");
        }
        const int32_t requested_keep = std::max(
                1,
                static_cast<int32_t>(std::ceil(span_patches * config.vision_token_keep_ratio)));
        const int32_t keep_count = std::min(requested_keep, static_cast<int32_t>(candidates.size()));
        span_plans.push_back({ score_offset, span.patch_start, keep_count, std::move(candidates) });
        result.selected_tiles_by_span.push_back(selected_tiles);
        result.stats.vision_tiles += static_cast<int32_t>(tiles.size());
        result.stats.vision_selected_tiles += static_cast<int32_t>(selected_tiles.size());
        result.stats.candidate_patches += static_cast<int32_t>(span_plans.back().candidates.size());
        result.stats.vision_kept_patches += keep_count;
        score_offset += span_patches;
    }

    std::vector<int32_t> fixed_indices;
    fixed_indices.reserve(2 * layout.spans.size() + layout.prefix_length - layout.prompt_start);
    for (const d2f_visual_span & span : layout.spans) {
        fixed_indices.push_back(span.token_start);
        fixed_indices.push_back(span.token_end - 1);
    }
    for (int32_t token = layout.prompt_start; token < layout.prefix_length; ++token) {
        fixed_indices.push_back(token);
    }

    result.keep_count = result.stats.vision_kept_patches + static_cast<int32_t>(fixed_indices.size());
    result.stats.cached_prefix_tokens = result.keep_count;
    result.stats.compression_ratio =
            static_cast<float>(result.keep_count) / result.stats.dense_prefix_tokens;
    result.source_indices.reserve(
            static_cast<size_t>(num_model_layers) * num_kv_heads * result.keep_count);

    for (int32_t model_layer = 0; model_layer < num_model_layers; ++model_layer) {
        const d2f_visual_layer_scores & layer = pooled_scores[
                nearest_scored_layer(pooled_scores, model_layer)];
        for (int32_t head = 0; head < num_kv_heads; ++head) {
            std::vector<int32_t> keep = fixed_indices;
            keep.reserve(result.keep_count);
            for (const span_plan & span : span_plans) {
                std::vector<int32_t> ranked = span.candidates;
                std::sort(ranked.begin(), ranked.end(), [&](int32_t left, int32_t right) {
                    const float left_score = layer.values[
                            static_cast<size_t>(head) * total_patches + span.score_start + left];
                    const float right_score = layer.values[
                            static_cast<size_t>(head) * total_patches + span.score_start + right];
                    return left_score != right_score ? left_score > right_score : left < right;
                });
                ranked.resize(span.keep_count);
                for (int32_t patch : ranked) {
                    keep.push_back(span.patch_start + patch);
                }
            }
            std::sort(keep.begin(), keep.end());
            if (static_cast<int32_t>(keep.size()) != result.keep_count ||
                std::adjacent_find(keep.begin(), keep.end()) != keep.end()) {
                throw std::logic_error("visual KV keep indices are not a shared-size set");
            }
            result.source_indices.insert(result.source_indices.end(), keep.begin(), keep.end());
        }
    }

    // Cache destinations are compact token ordinals, while their RoPE/D2F
    // positions must remain identical to the dense prefix. Every token in an
    // image span shares that image's position; prompt positions stay
    // sequential from prompt_position. The keep set can differ per KV head,
    // but its j-th entry always belongs to the same span (or prompt offset), so
    // one destination position vector is valid for all layers and heads.
    result.destination_positions.reserve(result.keep_count);
    const size_t canonical_offset = 0;
    for (int32_t destination = 0; destination < result.keep_count; ++destination) {
        const int32_t source = result.source_indices.at(canonical_offset + destination);
        bool mapped = false;
        for (const d2f_visual_span & span : layout.spans) {
            if (source >= span.token_start && source < span.token_end) {
                result.destination_positions.push_back(span.position);
                mapped = true;
                break;
            }
        }
        if (!mapped && source >= layout.prompt_start && source < layout.prefix_length) {
            result.destination_positions.push_back(
                    layout.prompt_position + source - layout.prompt_start);
            mapped = true;
        }
        if (!mapped) {
            throw std::logic_error("visual KV source token is outside the prefix layout");
        }
    }
    return result;
}

inline d2f_visual_kv_keep_plan build_visual_kv_keep_plan_from_qk(
        int32_t num_model_layers,
        const d2f_visual_prefix_layout & layout,
        const std::vector<d2f_post_rope_qk_view> & qk_views,
        const d2f_visual_kv_compression_config & config = {}) {
    validate_visual_kv_compression_config(config);
    std::vector<d2f_visual_layer_scores> scores;
    scores.reserve(qk_views.size());
    for (const d2f_post_rope_qk_view & view : qk_views) {
        scores.push_back(compute_visual_attention_scores(
                view,
                layout,
                config.vision_score_query_window));
    }
    return build_visual_kv_keep_plan(num_model_layers, layout, std::move(scores), config);
}

} // namespace lladao::detail
