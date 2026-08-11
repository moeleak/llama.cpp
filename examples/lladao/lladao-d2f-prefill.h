#pragma once

#include "lladao-d2f-engine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lladao::detail {

enum class prefix_chunk_kind : int32_t {
    complete,
    image,
    prompt,
};

struct prefix_chunk {
    int32_t offset = 0;
    int32_t length = 0;
    prefix_chunk_kind kind = prefix_chunk_kind::complete;
    int32_t component = -1;
};

inline const char * prefix_prefill_mode_name(d2f_prefix_prefill_mode mode) {
    switch (mode) {
        case d2f_prefix_prefill_mode::exact:           return "exact";
        case d2f_prefix_prefill_mode::component_exact: return "component_exact";
        case d2f_prefix_prefill_mode::packed_image:    return "packed_image";
        case d2f_prefix_prefill_mode::component_parallel: return "component_parallel";
    }
    return "invalid";
}

inline bool prefix_prefill_mode_valid(d2f_prefix_prefill_mode mode) {
    return mode == d2f_prefix_prefill_mode::exact ||
           mode == d2f_prefix_prefill_mode::component_exact ||
           mode == d2f_prefix_prefill_mode::packed_image ||
           mode == d2f_prefix_prefill_mode::component_parallel;
}

inline std::vector<prefix_chunk> plan_prefix_chunks(
        const std::vector<int32_t> & image_lengths,
        int32_t prompt_length,
        d2f_prefix_prefill_mode mode,
        int32_t pack_size) {
    if (!prefix_prefill_mode_valid(mode) || prompt_length <= 0 || pack_size <= 0) {
        throw std::invalid_argument("invalid prefix prefill configuration");
    }

    int64_t total_length = prompt_length;
    for (int32_t image_length : image_lengths) {
        if (image_length <= 0) {
            throw std::invalid_argument("image prefix spans must not be empty");
        }
        total_length += image_length;
    }
    if (total_length > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("prefix length exceeds int32 range");
    }

    if (mode == d2f_prefix_prefill_mode::exact) {
        return {{ 0, static_cast<int32_t>(total_length), prefix_chunk_kind::complete, -1 }};
    }

    const int32_t image_length = static_cast<int32_t>(total_length - prompt_length);
    if (mode == d2f_prefix_prefill_mode::component_parallel) {
        return {
            { 0, image_length, prefix_chunk_kind::image, -1 },
            { image_length, prompt_length, prefix_chunk_kind::prompt, -1 },
        };
    }

    std::vector<prefix_chunk> chunks;
    int32_t offset = 0;
    for (int32_t component = 0; component < static_cast<int32_t>(image_lengths.size()); ++component) {
        const int32_t image_length = image_lengths[component];
        if (mode == d2f_prefix_prefill_mode::component_exact) {
            chunks.push_back({ offset, image_length, prefix_chunk_kind::image, component });
        } else {
            int32_t component_offset = 0;
            while (component_offset < image_length) {
                const int32_t length = std::min(pack_size, image_length - component_offset);
                chunks.push_back({ offset + component_offset, length, prefix_chunk_kind::image, component });
                component_offset += length;
            }
        }
        offset += image_length;
    }
    chunks.push_back({ offset, prompt_length, prefix_chunk_kind::prompt, -1 });
    return chunks;
}

inline int32_t prefix_chunk_ubatch(
        const std::vector<prefix_chunk> & chunks,
        int32_t generation_length) {
    int32_t result = generation_length;
    for (const prefix_chunk & chunk : chunks) {
        result = std::max(result, chunk.length);
    }
    return result;
}

inline const char * prefix_chunk_kind_name(prefix_chunk_kind kind) {
    switch (kind) {
        case prefix_chunk_kind::complete: return "complete";
        case prefix_chunk_kind::image:    return "image";
        case prefix_chunk_kind::prompt:   return "prompt";
    }
    return "invalid";
}

inline bool vision_context_needs_rebuild(
        bool context_present,
        bool context_exact,
        bool requested_exact) {
    return !context_present || context_exact != requested_exact;
}

template<typename Context>
bool release_vision_context(Context & context, bool enabled) {
    if (!enabled || !context) {
        return false;
    }
    context.reset();
    return true;
}

} // namespace lladao::detail
