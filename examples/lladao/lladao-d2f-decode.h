#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lladao::detail {

struct d2f_generation_decode_plan {
    int32_t first_position           = 0;
    int32_t active_start             = 0;
    int32_t active_end               = 0;
    int32_t next_cached_complete_end = 0;
    std::vector<int32_t> logit_positions;

    int32_t input_rows() const {
        return active_end - first_position;
    }

    int32_t active_rows() const {
        return active_end - active_start;
    }

    int32_t rebuild_rows() const {
        return active_start - first_position;
    }

    bool requests_logits(int32_t position) const {
        return std::binary_search(logit_positions.begin(), logit_positions.end(), position);
    }

    int32_t batch_index(int32_t position, int32_t prefix_rows = 0) const {
        if (position < first_position || position >= active_end) {
            throw std::out_of_range("D2F generation position is outside the decode plan");
        }
        return prefix_rows + position - first_position;
    }
};

inline d2f_generation_decode_plan plan_d2f_generation_decode(
        const std::vector<int32_t> & tokens,
        int32_t mask_token_id,
        int32_t active_start,
        int32_t active_end,
        int32_t block_length,
        int32_t cached_complete_end,
        bool generation_block_cache,
        bool sparse_logits) {
    if (mask_token_id < 0 || active_start < 0 || active_end <= active_start || block_length <= 0 ||
        active_end > static_cast<int32_t>(tokens.size()) ||
        cached_complete_end < 0 || cached_complete_end > active_start ||
        active_start % block_length != 0 || active_end % block_length != 0 ||
        cached_complete_end % block_length != 0) {
        throw std::invalid_argument("invalid D2F generation decode range");
    }

    d2f_generation_decode_plan result;
    // A newly completed block is replayed with final tokens before this boundary advances.
    result.first_position = generation_block_cache ? cached_complete_end : 0;
    result.active_start = active_start;
    result.active_end = active_end;
    result.next_cached_complete_end = generation_block_cache ? active_start : 0;

    for (int32_t position = result.first_position; position < active_start; ++position) {
        if (tokens[position] == mask_token_id) {
            throw std::logic_error("cached D2F leading block still contains a mask token");
        }
    }

    if (sparse_logits) {
        for (int32_t position = active_start; position < active_end; ++position) {
            if (tokens[position] == mask_token_id) {
                result.logit_positions.push_back(position);
            }
        }
    } else {
        result.logit_positions.reserve(result.input_rows());
        for (int32_t position = result.first_position; position < active_end; ++position) {
            result.logit_positions.push_back(position);
        }
    }

    if (result.logit_positions.empty()) {
        throw std::logic_error("active D2F suffix has no requested logits rows");
    }
    return result;
}

struct d2f_decode_counters {
    uint64_t iterations   = 0;
    uint64_t input_rows   = 0;
    uint64_t active_rows  = 0;
    uint64_t rebuild_rows = 0;
    uint64_t logit_rows   = 0;
    uint64_t reused_input_rows = 0;

    void add(const d2f_generation_decode_plan & plan, int32_t prefix_rows = 0) {
        if (prefix_rows < 0) {
            throw std::invalid_argument("D2F prefix row count must not be negative");
        }
        ++iterations;
        input_rows += static_cast<uint64_t>(prefix_rows + plan.input_rows());
        active_rows += static_cast<uint64_t>(plan.active_rows());
        rebuild_rows += static_cast<uint64_t>(plan.rebuild_rows());
        logit_rows += plan.logit_positions.size();
        reused_input_rows += static_cast<uint64_t>(plan.first_position);
    }
};

} // namespace lladao::detail
