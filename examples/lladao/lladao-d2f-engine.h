#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace lladao {

struct d2f_engine_params {
    std::string model_path;
    std::string mmproj_path;
    std::string adapter_path;
    int32_t context_size = 16384;
    int32_t gpu_layers = 999;
    int32_t threads = 0;
    int32_t max_iterations = 256;
    int32_t mask_token_id = -1;
    int32_t yarn_original_context = 16384;
    float yarn_factor = 1.0f;
    float adapter_scale = 1.0f;
    int8_t vision_gpu = -1;
    bool prefix_cache = true;
    bool print_timings = true;
};

struct d2f_request {
    std::string image_path;
    std::string prompt;
    std::string retrieval_query;
    int32_t full_page_tile_size = 980;
    int32_t tile_retrieval_topk = 0;
    int32_t tile_retrieval_mask_rounds = 2;
    bool full_page_tiles = false;
    bool full_page_overview = true;
    bool preprocess_only = false;
};

struct d2f_result {
    std::string text;
    int32_t iterations = 0;
    int32_t image_tokens = 0;
    int32_t input_tokens = 0;
    uint64_t state_bytes = 0;
    double prefill_seconds = 0.0;
    double state_io_seconds = 0.0;
    double decode_seconds = 0.0;
    double generation_seconds = 0.0;
    bool cancelled = false;
    bool preprocessed_only = false;
    bool prefilled_only = false;
    bool decoded_from_state = false;
};

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
