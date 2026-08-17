#pragma once

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fast_dvlm {

struct speculative_step {
    int32_t                  accepted     = 0;
    int32_t                  block_start  = 0;
    int32_t                  output_count = 0;
    llama_token              inherited    = LLAMA_TOKEN_NULL;
    std::vector<llama_token> output;
};

template <typename Argmax>
void apply_draft_predictions(std::vector<llama_token> & tokens,
                             llama_token                mask_token,
                             int32_t                    token_shift,
                             Argmax &&                  argmax) {
    if (tokens.empty() || token_shift < 0) {
        throw std::invalid_argument("invalid Fast-dVLM draft shape or token shift");
    }

    for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
        if (tokens[i] != mask_token) {
            continue;
        }
        const int32_t logits_row = token_shift > 0 ? std::max<int32_t>(0, i - token_shift) : i;
        tokens[i]                = argmax(logits_row);
    }
}

inline speculative_step resolve_speculative_step(const std::vector<llama_token> & drafted,
                                                 const std::vector<llama_token> & causal_next,
                                                 int32_t                          block_start) {
    if (drafted.empty() || drafted.size() != causal_next.size()) {
        throw std::invalid_argument("Fast-dVLM draft and causal verification sizes differ");
    }
    if (block_start < 0 || block_start > static_cast<int32_t>(drafted.size())) {
        throw std::invalid_argument("Fast-dVLM block start is outside the draft");
    }

    speculative_step result;
    result.block_start = block_start;

    for (int32_t i = 0; i + 1 < static_cast<int32_t>(drafted.size()); ++i) {
        if (causal_next[i] != drafted[i + 1]) {
            break;
        }
        ++result.accepted;
    }

    // The causal token at the first mismatch (or after a fully matching
    // draft) is inherited by the following request, matching SGLang's
    // SpeculativeBlock implementation.
    ++result.accepted;
    result.accepted     = std::min<int32_t>(result.accepted, drafted.size());
    result.output_count = std::max<int32_t>(result.accepted - block_start, 0);
    result.output.assign(drafted.begin() + block_start, drafted.begin() + block_start + result.output_count);
    result.inherited = causal_next[result.accepted - 1];
    return result;
}

}  // namespace fast_dvlm
