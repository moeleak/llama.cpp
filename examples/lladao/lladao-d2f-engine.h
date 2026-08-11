#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>

namespace lladao {

enum class d2f_prefix_prefill_mode : int32_t {
    exact = 0,
    component_exact = 1,
    packed_image = 2,
    component_parallel = 3,
};

enum class d2f_flash_attention_mode : int32_t {
    auto_select = -1,
    disabled = 0,
    enabled = 1,
};

struct d2f_engine_params {
    std::string model_path;
    std::string mmproj_path;
    std::string adapter_path;
    int32_t context_size = 16384;
    int32_t gpu_layers = 999;
    int32_t threads = 0;
    uint64_t cpu_mask = 0;
    uint32_t cpu_poll = 50;
    int32_t max_iterations = 256;
    int32_t mask_token_id = -1;
    int32_t exact_image_max_edge = 0;
    d2f_prefix_prefill_mode prefix_prefill_mode = d2f_prefix_prefill_mode::exact;
    int32_t prefix_pack_size = 512;
    int32_t vision_kv_tile_size = 16;
    int32_t vision_kv_topk_tiles = 20;
    int32_t vision_kv_query_window = 32;
    int32_t vision_kv_score_layers = 4;
    int32_t vision_kv_pool_kernel = 7;
    float vision_kv_keep_ratio = 0.75f;
    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;
    d2f_flash_attention_mode flash_attention_mode = d2f_flash_attention_mode::auto_select;
    int32_t yarn_original_context = 16384;
    float yarn_factor = 1.0f;
    float adapter_scale = 1.0f;
    int8_t vision_gpu = -1;
    bool prefix_cache = true;
    bool generation_block_cache = false;
    bool sparse_generation_logits = false;
    bool cpu_strict = false;
    bool release_vision_after_encode = false;
    bool vision_kv_compression = false;
    bool print_timings = true;
};

struct d2f_request {
    std::string image_path;
    // Stable content identity supplied by the caller. Reusing this key allows a
    // planner and grounding request over the exact same screenshot to share the
    // vision-tower output even when a language LoRA is switched between them.
    std::string image_cache_key;
    std::string prompt;
    std::string retrieval_query;
    int32_t full_page_tile_size = 980;
    int32_t tile_retrieval_topk = 0;
    int32_t tile_retrieval_mask_rounds = 2;
    bool full_page_tiles = false;
    bool full_page_overview = true;
    bool exact_image = false;
    bool preprocess_only = false;
};

struct d2f_result {
    std::string text;
    std::string language_device_mode;
    std::string cache_type_k;
    std::string cache_type_v;
    std::string flash_attention_mode;
    std::string flash_attention_requested;
    std::string flash_attention_resolved;
    std::string prefix_prefill_mode;
    std::string cpu_mask_requested;
    std::string cpu_mask_resolved;
    int32_t prefix_pack_size = 0;
    int32_t prefix_chunk_count = 0;
    int32_t image_span_count = 0;
    int32_t image_prefill_calls = 0;
    int32_t dense_prefix_tokens = 0;
    int32_t cached_prefix_tokens = 0;
    int32_t vision_patches = 0;
    int32_t vision_kept_patches = 0;
    int32_t vision_tiles = 0;
    int32_t vision_selected_tiles = 0;
    int32_t cpu_threads_requested = 0;
    int32_t cpu_threads_resolved = 0;
    uint32_t cpu_poll_requested = 0;
    uint32_t cpu_poll_resolved = 0;
    int32_t iterations = 0;
    int32_t image_tokens = 0;
    int32_t input_tokens = 0;
    uint64_t state_bytes = 0;
    uint64_t d2f_input_rows = 0;
    uint64_t d2f_active_rows = 0;
    uint64_t d2f_rebuild_rows = 0;
    uint64_t d2f_logit_rows = 0;
    uint64_t d2f_reused_input_rows = 0;
    double prefill_seconds = 0.0;
    double state_io_seconds = 0.0;
    double decode_seconds = 0.0;
    double vision_encode_seconds = 0.0;
    double kv_cache_compression_seconds = 0.0;
    double generation_seconds = 0.0;
    bool vision_cache_hit = false;
    bool generation_block_cache = false;
    bool sparse_generation_logits = false;
    bool cpu_strict_requested = false;
    bool cpu_strict_resolved = false;
    bool cpu_threadpool_attached = false;
    bool vision_threadpool_shared = false;
    bool release_vision_after_encode = false;
    bool vision_context_released = false;
    bool vision_kv_compression = false;
    float vision_kv_compression_ratio = 1.0f;
    bool cancelled = false;
    bool preprocessed_only = false;
    bool prefilled_only = false;
    bool decoded_from_state = false;
};

namespace detail {

uint64_t parse_cpu_mask_hex(const char * value);
std::string format_cpu_mask(uint64_t mask);
int32_t resolve_cpu_threads(uint64_t mask, int32_t requested_threads);
const char * language_device_mode(int32_t gpu_layers);

} // namespace detail

class d2f_engine {
  public:
    explicit d2f_engine(const d2f_engine_params & params);
    ~d2f_engine();

    d2f_engine(const d2f_engine &) = delete;
    d2f_engine & operator=(const d2f_engine &) = delete;
    d2f_engine(d2f_engine &&) noexcept;
    d2f_engine & operator=(d2f_engine &&) noexcept;

    d2f_result generate(const d2f_request & request);
    d2f_result prefill_to_file(const d2f_request & request, const std::string & state_path);
    d2f_result decode_from_file(const std::string & state_path);

    void set_adapter(const std::string & path, float scale = 1.0f);
    void clear_adapter();
    void cancel() noexcept;
    bool busy() const noexcept;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

int cli_main(int argc, char ** argv);

} // namespace lladao
