#pragma once

#include <cstdint>
#include <vector>

struct diffusion_d2f_scheduler_params {
    int32_t generation_length       = 64;
    int32_t block_length            = 16;
    int32_t max_iterations          = 256;
    int32_t mask_token_id           = -1;
    int32_t initial_token_id        = -1;
    int32_t eos_token_id            = -1;
    float   block_add_threshold     = 0.1f;
    float   decoded_token_threshold = 0.95f;
    float   skip_threshold          = 0.9f;
};

struct diffusion_d2f_candidate {
    int32_t token_id   = -1;
    float   confidence = 0.0f;
    bool    valid      = false;
};

struct diffusion_d2f_update {
    int32_t position   = -1;
    int32_t token_id   = -1;
    float   confidence = 0.0f;
};

struct diffusion_d2f_step {
    int32_t iteration        = 0;
    int32_t active_start     = 0;
    int32_t active_end       = 0;
    int32_t blocks_added     = 0;
    int32_t blocks_completed = 0;
    bool    done             = false;
};

class diffusion_d2f_scheduler {
  public:
    explicit diffusion_d2f_scheduler(const diffusion_d2f_scheduler_params & params);

    diffusion_d2f_step                prepare_step();
    std::vector<diffusion_d2f_update> apply_candidates(const std::vector<diffusion_d2f_candidate> & candidates);

    const std::vector<int32_t> & tokens() const;
    int32_t                      output_length() const;
    bool                         done() const;

  private:
    struct block_state {
        int32_t masks = 0;
        int32_t total = 0;
    };

    float block_progress(int32_t block) const;
    bool  has_mask_before(int32_t end) const;
    void  add_block();
    void  update_done();

    diffusion_d2f_scheduler_params params_;
    std::vector<int32_t>           tokens_;
    std::vector<block_state>       blocks_;
    std::vector<bool>              ready_blocks_;
    int32_t                        blocks_completed_ = 0;
    int32_t                        eos_position_     = -1;
    int32_t                        iteration_        = 0;
    bool                           step_prepared_    = false;
    bool                           done_             = false;
};

std::vector<diffusion_d2f_candidate> diffusion_d2f_argmax_candidates(const float * logits,
                                                                     int32_t       n_rows,
                                                                     int32_t       n_vocab,
                                                                     int32_t       first_logit_position,
                                                                     int32_t       first_target_position,
                                                                     int32_t       n_targets,
                                                                     bool          shift_logits);

diffusion_d2f_candidate diffusion_d2f_argmax_candidate(const float * logits,
                                                       int32_t       n_vocab);

bool diffusion_d2f_attention_allowed(int32_t query_position,
                                     int32_t key_position,
                                     int32_t prompt_length,
                                     int32_t block_length);
