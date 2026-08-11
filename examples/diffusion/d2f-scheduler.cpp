#include "d2f-scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

static bool threshold_valid(float threshold) {
    return threshold >= 0.0f && threshold <= 1.0f;
}

diffusion_d2f_scheduler::diffusion_d2f_scheduler(const diffusion_d2f_scheduler_params & params) :
    params_(params),
    tokens_(params.generation_length, params.mask_token_id) {
    if (params_.generation_length <= 0 || params_.block_length < 2 ||
        params_.generation_length % params_.block_length != 0) {
        throw std::invalid_argument("D2F generation length must be a positive block multiple");
    }
    if (params_.max_iterations <= 0) {
        throw std::invalid_argument("D2F max iterations must be positive");
    }
    if (params_.mask_token_id < 0 || params_.initial_token_id < 0 ||
        params_.mask_token_id == params_.initial_token_id) {
        throw std::invalid_argument("D2F mask and initial token IDs must be distinct");
    }
    if (!threshold_valid(params_.block_add_threshold) || !threshold_valid(params_.decoded_token_threshold) ||
        !threshold_valid(params_.skip_threshold)) {
        throw std::invalid_argument("D2F thresholds must be in [0, 1]");
    }
}

float diffusion_d2f_scheduler::block_progress(int32_t block) const {
    const block_state & state = blocks_.at(block);
    return state.total > 0 ? 1.0f - static_cast<float>(state.masks) / state.total : 1.0f;
}

bool diffusion_d2f_scheduler::has_mask_before(int32_t end) const {
    end = std::min(end, static_cast<int32_t>(tokens_.size()));
    return std::find(tokens_.begin(), tokens_.begin() + end, params_.mask_token_id) != tokens_.begin() + end;
}

void diffusion_d2f_scheduler::add_block() {
    const int32_t block = static_cast<int32_t>(blocks_.size());
    if (block == 0) {
        tokens_[0] = params_.initial_token_id;
        blocks_.push_back({ params_.block_length - 1, params_.block_length - 1 });
    } else {
        blocks_.push_back({ params_.block_length, params_.block_length });
    }
}

void diffusion_d2f_scheduler::update_done() {
    if (eos_position_ >= 0 && !has_mask_before(eos_position_ + 1)) {
        done_ = true;
        return;
    }

    const int32_t max_blocks = params_.generation_length / params_.block_length;
    if (blocks_completed_ == static_cast<int32_t>(blocks_.size()) &&
        (static_cast<int32_t>(blocks_.size()) == max_blocks || eos_position_ >= 0)) {
        done_ = true;
    }
}

diffusion_d2f_step diffusion_d2f_scheduler::prepare_step() {
    if (step_prepared_) {
        throw std::logic_error("D2F step already prepared");
    }
    if (done_) {
        return { iteration_, output_length(), output_length(), static_cast<int32_t>(blocks_.size()), blocks_completed_,
                 true };
    }
    if (iteration_ >= params_.max_iterations) {
        throw std::runtime_error("D2F generation exceeded max iterations");
    }

    const int32_t max_blocks = params_.generation_length / params_.block_length;
    if (blocks_.empty()) {
        add_block();
    } else if (static_cast<int32_t>(blocks_.size()) < max_blocks && eos_position_ < 0 &&
               block_progress(static_cast<int32_t>(blocks_.size()) - 1) >= params_.block_add_threshold) {
        add_block();
    }

    while (blocks_completed_ < static_cast<int32_t>(blocks_.size()) && blocks_[blocks_completed_].masks == 0) {
        ++blocks_completed_;
    }

    update_done();
    ++iteration_;
    if (done_) {
        return { iteration_, output_length(), output_length(), static_cast<int32_t>(blocks_.size()), blocks_completed_,
                 true };
    }

    ready_blocks_.assign(blocks_.size(), false);
    ready_blocks_[blocks_completed_] = true;
    for (int32_t block = blocks_completed_ + 1; block < static_cast<int32_t>(blocks_.size()); ++block) {
        ready_blocks_[block] = block_progress(block - 1) >= params_.decoded_token_threshold;
    }

    step_prepared_ = true;
    return {
        iteration_,
        blocks_completed_ * params_.block_length,
        static_cast<int32_t>(blocks_.size()) * params_.block_length,
        static_cast<int32_t>(blocks_.size()),
        blocks_completed_,
        false,
    };
}

std::vector<diffusion_d2f_update> diffusion_d2f_scheduler::apply_candidates(
    const std::vector<diffusion_d2f_candidate> & candidates) {
    if (!step_prepared_) {
        throw std::logic_error("D2F step is not prepared");
    }
    if (candidates.size() != tokens_.size()) {
        throw std::invalid_argument("D2F candidate count must match generation length");
    }

    std::vector<diffusion_d2f_update> updates;
    for (int32_t block = blocks_completed_; block < static_cast<int32_t>(blocks_.size()); ++block) {
        const int32_t        start = block * params_.block_length;
        const int32_t        end   = start + params_.block_length;
        std::vector<int32_t> selected;
        int32_t              fallback = -1;

        for (int32_t position = start; position < end; ++position) {
            if (tokens_[position] != params_.mask_token_id || !candidates[position].valid) {
                continue;
            }
            if (fallback < 0 || candidates[position].confidence > candidates[fallback].confidence) {
                fallback = position;
            }
            if (candidates[position].confidence >= params_.skip_threshold) {
                selected.push_back(position);
            }
        }

        if (selected.empty() && ready_blocks_[block] && fallback >= 0) {
            selected.push_back(fallback);
        }

        for (int32_t position : selected) {
            const diffusion_d2f_candidate & candidate = candidates[position];
            tokens_[position]                         = candidate.token_id;
            --blocks_[block].masks;
            updates.push_back({ position, candidate.token_id, candidate.confidence });
            if (candidate.token_id == params_.eos_token_id && (eos_position_ < 0 || position < eos_position_)) {
                eos_position_ = position;
            }
        }
    }

    step_prepared_ = false;
    if (updates.empty()) {
        throw std::runtime_error("D2F scheduler made no progress");
    }
    return updates;
}

const std::vector<int32_t> & diffusion_d2f_scheduler::tokens() const {
    return tokens_;
}

int32_t diffusion_d2f_scheduler::output_length() const {
    if (eos_position_ >= 0) {
        return eos_position_ + 1;
    }
    return static_cast<int32_t>(blocks_.size()) * params_.block_length;
}

bool diffusion_d2f_scheduler::done() const {
    return done_;
}

std::vector<diffusion_d2f_candidate> diffusion_d2f_argmax_candidates(const float * logits,
                                                                     int32_t       n_rows,
                                                                     int32_t       n_vocab,
                                                                     int32_t       first_logit_position,
                                                                     int32_t       first_target_position,
                                                                     int32_t       n_targets,
                                                                     bool          shift_logits) {
    if (!logits || n_rows < 0 || n_vocab <= 0 || n_targets < 0) {
        throw std::invalid_argument("invalid D2F logits view");
    }

    std::vector<diffusion_d2f_candidate> result(n_targets);
    for (int32_t target = 0; target < n_targets; ++target) {
        const int32_t target_position = first_target_position + target;
        const int32_t source_position = target_position - (shift_logits ? 1 : 0);
        const int32_t row             = source_position - first_logit_position;
        if (row < 0 || row >= n_rows) {
            continue;
        }

        result[target] = diffusion_d2f_argmax_candidate(
                logits + static_cast<size_t>(row) * n_vocab,
                n_vocab);
    }
    return result;
}

diffusion_d2f_candidate diffusion_d2f_argmax_candidate(
        const float * logits,
        int32_t n_vocab) {
    if (!logits || n_vocab <= 0) {
        throw std::invalid_argument("invalid D2F logits row");
    }

    int32_t token = 0;
    for (int32_t id = 1; id < n_vocab; ++id) {
        if (logits[id] > logits[token]) {
            token = id;
        }
    }

    const float maximum = logits[token];
    if (!std::isfinite(maximum)) {
        return {};
    }
    double denominator = 0.0;
    for (int32_t id = 0; id < n_vocab; ++id) {
        denominator += std::exp(static_cast<double>(logits[id] - maximum));
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        return {};
    }
    return { token, static_cast<float>(1.0 / denominator), true };
}

bool diffusion_d2f_attention_allowed(int32_t query_position,
                                     int32_t key_position,
                                     int32_t prompt_length,
                                     int32_t block_length) {
    if (query_position < 0 || key_position < 0 || prompt_length < 0 || block_length <= 0) {
        return false;
    }
    if (query_position < prompt_length) {
        return key_position < prompt_length;
    }
    if (key_position < prompt_length) {
        return true;
    }

    const int32_t query_block = (query_position - prompt_length) / block_length;
    const int32_t key_block   = (key_position - prompt_length) / block_length;
    return key_block <= query_block;
}
