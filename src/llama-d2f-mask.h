#pragma once

#include <cstdint>

static inline bool llm_d2f_attention_visible(
        int32_t query_index,
        int32_t query_position,
        int32_t key_index,
        int32_t key_position,
        int32_t image_prefix_length,
        int32_t prefix_length,
        int32_t block_size) {
    if (query_index < image_prefix_length) {
        return key_index < image_prefix_length && key_position == query_position;
    }

    if (query_index < prefix_length) {
        return key_index < prefix_length;
    }

    if (key_index < prefix_length) {
        return true;
    }

    const int32_t query_block = (query_index - prefix_length) / block_size;
    const int32_t key_block   = (key_index - prefix_length) / block_size;
    return key_block <= query_block;
}

static inline bool llm_d2f_attention_visible_position(
        int32_t query_position,
        int32_t key_position,
        int32_t prompt_position,
        int32_t generation_position,
        int32_t block_size) {
    if (query_position < prompt_position) {
        return key_position < prompt_position && key_position == query_position;
    }

    if (query_position < generation_position) {
        return key_position < generation_position;
    }

    if (key_position < generation_position) {
        return true;
    }

    const int32_t query_block = (query_position - generation_position) / block_size;
    const int32_t key_block   = (key_position - generation_position) / block_size;
    return key_block <= query_block;
}
