#pragma once

#include "lladao-d2f-engine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
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

struct prefix_parallel_audit {
    std::vector<int32_t> cu_seqlens;
    int64_t attention_pairs_dense = 0;
    int64_t attention_pairs_packed = 0;
    int32_t domains = 0;
    int32_t image_decode_calls = 0;
    bool flash_attention_resolved = false;
    bool batched_submission = false;
};

struct language_context_capacity {
    int32_t resident = 0;
    int32_t batch = 0;
    int32_t outputs = 0;
    int32_t sequences = 0;
};

template<typename Enable, typename Clear, typename Body>
decltype(auto) with_packed_prefill_contract(
        Enable && enable,
        Clear && clear,
        Body && body) {
    enable();

    using clear_type = typename std::remove_reference<Clear>::type;
    static_assert(noexcept(std::declval<clear_type &>()()),
            "packed prefill clear callback must be noexcept");
    struct clear_guard {
        clear_type * clear;

        ~clear_guard() noexcept {
            (*clear)();
        }
    } guard { &clear };

    return std::forward<Body>(body)();
}

inline bool language_context_needs_rebuild(
        bool context_present,
        const language_context_capacity & capacity,
        int32_t required_resident,
        int32_t required_batch,
        int32_t required_outputs,
        int32_t required_sequences) {
    return !context_present ||
           required_resident > capacity.resident ||
           required_batch > capacity.batch ||
           required_outputs > capacity.outputs ||
           required_sequences > capacity.sequences;
}

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

inline prefix_parallel_audit audit_prefix_parallelism(
        const std::vector<int32_t> & image_lengths,
        d2f_prefix_prefill_mode mode,
        bool flash_attention_resolved,
        int32_t pack_size = 512) {
    if (image_lengths.empty()) {
        throw std::invalid_argument("parallel prefix audit requires at least one image span");
    }

    prefix_parallel_audit result;
    result.cu_seqlens.reserve(image_lengths.size() + 1);
    result.cu_seqlens.push_back(0);
    int64_t total_length = 0;
    for (int32_t image_length : image_lengths) {
        if (image_length <= 0) {
            throw std::invalid_argument("image prefix spans must not be empty");
        }
        total_length += image_length;
        if (total_length > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error("prefix length exceeds int32 range");
        }
        result.cu_seqlens.push_back(static_cast<int32_t>(total_length));
        result.attention_pairs_packed += static_cast<int64_t>(image_length) * image_length;
    }
    result.attention_pairs_dense = total_length * total_length;
    result.domains = static_cast<int32_t>(image_lengths.size());
    result.flash_attention_resolved = flash_attention_resolved;

    const std::vector<prefix_chunk> chunks = plan_prefix_chunks(image_lengths, 1, mode, pack_size);
    result.image_decode_calls = static_cast<int32_t>(std::count_if(
            chunks.begin(),
            chunks.end(),
            [](const prefix_chunk & chunk) {
                return chunk.kind == prefix_chunk_kind::image;
            }));
    result.batched_submission =
            mode == d2f_prefix_prefill_mode::component_parallel &&
            result.domains > 1 &&
            result.image_decode_calls == 1;
    return result;
}

inline void validate_prefix_parallel_backend(
        d2f_prefix_prefill_mode mode,
        bool flash_attention_resolved) {
    if (mode == d2f_prefix_prefill_mode::component_parallel && !flash_attention_resolved) {
        throw std::invalid_argument("component_parallel prefill requires resolved Flash Attention");
    }
}

inline int32_t prefix_attention_domain(
        const prefix_parallel_audit & audit,
        int32_t token_index) {
    if (audit.cu_seqlens.size() < 2 || token_index < 0 || token_index >= audit.cu_seqlens.back()) {
        throw std::out_of_range("packed prefix token index is out of range");
    }
    const auto boundary = std::upper_bound(
            audit.cu_seqlens.begin(), audit.cu_seqlens.end(), token_index);
    return static_cast<int32_t>(boundary - audit.cu_seqlens.begin() - 1);
}

inline bool prefix_attention_allowed(
        const prefix_parallel_audit & audit,
        int32_t query_index,
        int32_t key_index) {
    return prefix_attention_domain(audit, query_index) ==
           prefix_attention_domain(audit, key_index);
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
