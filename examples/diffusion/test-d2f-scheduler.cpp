#include "d2f-scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#define CHECK(condition)                                           \
    do {                                                           \
        if (!(condition)) {                                        \
            throw std::runtime_error("check failed: " #condition); \
        }                                                          \
    } while (0)

static diffusion_d2f_scheduler_params test_params() {
    diffusion_d2f_scheduler_params params;
    params.mask_token_id    = 99;
    params.initial_token_id = 1;
    params.eos_token_id     = 2;
    return params;
}

static std::vector<diffusion_d2f_candidate> empty_candidates(int32_t length) {
    return std::vector<diffusion_d2f_candidate>(length);
}

static void set_candidate(std::vector<diffusion_d2f_candidate> & candidates,
                          int32_t                                position,
                          int32_t                                token,
                          float                                  confidence) {
    candidates[position] = { token, confidence, true };
}

static void test_shifted_logits() {
    const std::vector<float> logits = {
        0.0f, 1.0f,           0.0f, 0.0f, 0.0f, 2.0f,           std::log(8.0f), 0.0f, 0.0f,
        0.0f, std::log(8.0f), 0.0f, 0.0f, 0.0f, std::log(8.0f), std::log(8.0f), 0.0f, 0.0f,
    };

    const auto shifted = diffusion_d2f_argmax_candidates(logits.data(), 6, 3, 0, 3, 3, true);
    CHECK(shifted[0].token_id == 0);
    CHECK(shifted[1].token_id == 1);
    CHECK(shifted[2].token_id == 2);
    CHECK(std::fabs(shifted[0].confidence - 0.8f) < 1e-6f);

    const auto unshifted = diffusion_d2f_argmax_candidates(logits.data(), 6, 3, 0, 3, 3, false);
    CHECK(unshifted[0].token_id == 1);
    CHECK(unshifted[1].token_id == 2);
    CHECK(unshifted[2].token_id == 0);
}

static void test_block_causal_attention() {
    const int32_t prompt = 3;
    const int32_t block  = 4;

    for (int32_t query = 0; query < prompt; ++query) {
        for (int32_t key = 0; key < prompt; ++key) {
            CHECK(diffusion_d2f_attention_allowed(query, key, prompt, block));
        }
        CHECK(!diffusion_d2f_attention_allowed(query, prompt, prompt, block));
    }

    CHECK(diffusion_d2f_attention_allowed(prompt, prompt + block - 1, prompt, block));
    CHECK(diffusion_d2f_attention_allowed(prompt + block - 1, prompt, prompt, block));
    CHECK(!diffusion_d2f_attention_allowed(prompt, prompt + block, prompt, block));
    CHECK(diffusion_d2f_attention_allowed(prompt + block, prompt, prompt, block));
    CHECK(diffusion_d2f_attention_allowed(prompt + block, prompt + 2 * block - 1, prompt, block));
    CHECK(!diffusion_d2f_attention_allowed(prompt + block, prompt + 2 * block, prompt, block));
}

static void test_threshold_trace() {
    diffusion_d2f_scheduler scheduler(test_params());

    diffusion_d2f_step step = scheduler.prepare_step();
    CHECK(step.iteration == 1);
    CHECK(step.active_start == 0);
    CHECK(step.active_end == 16);
    CHECK(step.blocks_added == 1);
    CHECK(scheduler.tokens()[0] == 1);

    auto candidates = empty_candidates(64);
    for (int32_t position = 1; position < 16; ++position) {
        set_candidate(candidates, position, 100 + position, 0.5f);
    }
    set_candidate(candidates, 1, 101, 0.91f);
    auto updates = scheduler.apply_candidates(candidates);
    CHECK(updates.size() == 1);
    CHECK(updates[0].position == 1);

    step = scheduler.prepare_step();
    CHECK(step.blocks_added == 1);
    candidates = empty_candidates(64);
    for (int32_t position = 2; position < 16; ++position) {
        set_candidate(candidates, position, 100 + position, 0.5f);
    }
    set_candidate(candidates, 2, 102, 0.8f);
    updates = scheduler.apply_candidates(candidates);
    CHECK(updates.size() == 1);
    CHECK(updates[0].position == 2);

    step = scheduler.prepare_step();
    CHECK(step.blocks_added == 2);
    CHECK(step.active_end == 32);
    candidates = empty_candidates(64);
    for (int32_t position = 3; position < 32; ++position) {
        set_candidate(candidates, position, 100 + position, 0.4f);
    }
    set_candidate(candidates, 3, 103, 0.7f);
    set_candidate(candidates, 16, 116, 0.91f);
    updates = scheduler.apply_candidates(candidates);
    CHECK(updates.size() == 2);
    CHECK(updates[0].position == 3);
    CHECK(updates[1].position == 16);
}

static void test_synthetic_logits_trace() {
    diffusion_d2f_scheduler_params params = test_params();
    params.generation_length              = 8;
    params.block_length                   = 4;
    diffusion_d2f_scheduler scheduler(params);

    const int32_t      prompt_length = 5;
    const int32_t      n_vocab       = 3;
    std::vector<float> logits((prompt_length + params.generation_length) * n_vocab, 0.0f);

    diffusion_d2f_step step = scheduler.prepare_step();
    CHECK(step.blocks_added == 1);

    logits[prompt_length * n_vocab + 1] = std::log(20.0f);
    auto candidates = diffusion_d2f_argmax_candidates(logits.data(), prompt_length + params.generation_length, n_vocab,
                                                      0, prompt_length, params.generation_length, true);
    auto updates    = scheduler.apply_candidates(candidates);
    CHECK(updates.size() == 1);
    CHECK(updates[0].position == 1);
    CHECK(updates[0].token_id == 1);

    step = scheduler.prepare_step();
    CHECK(step.blocks_added == 2);
    std::fill(logits.begin(), logits.end(), 0.0f);
    logits[(prompt_length + 1) * n_vocab + 2] = std::log(20.0f);
    logits[(prompt_length + 3) * n_vocab + 1] = std::log(20.0f);
    candidates = diffusion_d2f_argmax_candidates(logits.data(), prompt_length + params.generation_length, n_vocab, 0,
                                                 prompt_length, params.generation_length, true);
    updates    = scheduler.apply_candidates(candidates);
    CHECK(updates.size() == 2);
    CHECK(updates[0].position == 2);
    CHECK(updates[0].token_id == 2);
    CHECK(updates[1].position == 4);
    CHECK(updates[1].token_id == 1);
}

static void test_ready_block_fallback() {
    diffusion_d2f_scheduler_params params = test_params();
    params.generation_length              = 8;
    params.block_length                   = 4;
    params.block_add_threshold            = 0.1f;
    params.decoded_token_threshold        = 0.5f;
    diffusion_d2f_scheduler scheduler(params);

    scheduler.prepare_step();
    auto candidates = empty_candidates(8);
    for (int32_t position = 1; position < 4; ++position) {
        set_candidate(candidates, position, 10 + position, 0.5f);
    }
    scheduler.apply_candidates(candidates);

    diffusion_d2f_step step = scheduler.prepare_step();
    CHECK(step.blocks_added == 2);
    candidates = empty_candidates(8);
    for (int32_t position = 2; position < 8; ++position) {
        set_candidate(candidates, position, 10 + position, 0.5f);
    }
    auto updates = scheduler.apply_candidates(candidates);
    CHECK(updates.size() == 1);
    CHECK(updates[0].position == 2);

    step       = scheduler.prepare_step();
    candidates = empty_candidates(8);
    for (int32_t position = 3; position < 8; ++position) {
        set_candidate(candidates, position, 10 + position, 0.5f);
    }
    updates = scheduler.apply_candidates(candidates);
    CHECK(updates.size() == 2);
    CHECK(updates[0].position == 3);
    CHECK(updates[1].position == 4);
}

static void test_eos_completion() {
    diffusion_d2f_scheduler_params params = test_params();
    params.generation_length              = 16;
    params.block_length                   = 8;
    diffusion_d2f_scheduler scheduler(params);

    scheduler.prepare_step();
    auto candidates = empty_candidates(16);
    for (int32_t position = 1; position < 8; ++position) {
        set_candidate(candidates, position, 20 + position, 0.5f);
    }
    set_candidate(candidates, 5, params.eos_token_id, 0.95f);
    scheduler.apply_candidates(candidates);

    while (!scheduler.done()) {
        diffusion_d2f_step step = scheduler.prepare_step();
        if (step.done) {
            break;
        }
        candidates = empty_candidates(16);
        for (int32_t position = step.active_start; position < step.active_end; ++position) {
            if (scheduler.tokens()[position] == params.mask_token_id) {
                set_candidate(candidates, position, 20 + position, position < 5 ? 0.95f : 0.5f);
            }
        }
        scheduler.apply_candidates(candidates);
    }

    CHECK(scheduler.done());
    CHECK(scheduler.output_length() == 6);
    CHECK(scheduler.tokens()[6] == params.mask_token_id);
}

int main() {
    test_shifted_logits();
    test_block_causal_attention();
    test_threshold_trace();
    test_synthetic_logits_trace();
    test_ready_block_fallback();
    test_eos_completion();
    std::cout << "D2F scheduler tests passed\n";
    return 0;
}
