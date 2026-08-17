#include "d2f-scheduler.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "lladao-d2f-decode.h"
#include "lladao-d2f-engine.h"
#include "lladao-d2f-kv-compression.h"
#include "lladao-d2f-prefill.h"
#include "lladao-d2f-qk-capture.h"
#include "llama-cpp.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd-image.h"
#include "mtmd.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

constexpr int32_t D2F_GENERATION_LENGTH = 64;
constexpr int32_t D2F_BLOCK_LENGTH      = 16;

void print_phase_resources(
        const char * phase,
        double wall_seconds,
        const lladao::d2f_phase_resource_usage & usage,
        const char * component = nullptr) {
    if (component) {
        std::fprintf(
                stderr,
                "D2F phase_resources phase=%s component=%s"
                " wall_seconds=%.6f cpu_seconds=%.6f"
                " minor_faults=%" PRIu64 " major_faults=%" PRIu64
                " inblock=%" PRIu64 " oublock=%" PRIu64
                " proc_io_available=%s read_bytes=%" PRIu64 " write_bytes=%" PRIu64 "\n",
                phase,
                component,
                wall_seconds,
                usage.process_cpu_seconds,
                usage.minor_page_faults,
                usage.major_page_faults,
                usage.block_input_operations,
                usage.block_output_operations,
                usage.proc_io_available ? "true" : "false",
                usage.read_bytes,
                usage.write_bytes);
    } else {
        std::fprintf(
                stderr,
                "D2F phase_resources phase=%s wall_seconds=%.6f cpu_seconds=%.6f"
                " minor_faults=%" PRIu64 " major_faults=%" PRIu64
                " inblock=%" PRIu64 " oublock=%" PRIu64
                " proc_io_available=%s read_bytes=%" PRIu64 " write_bytes=%" PRIu64 "\n",
                phase,
                wall_seconds,
                usage.process_cpu_seconds,
                usage.minor_page_faults,
                usage.major_page_faults,
                usage.block_input_operations,
                usage.block_output_operations,
                usage.proc_io_available ? "true" : "false",
                usage.read_bytes,
                usage.write_bytes);
    }
}

struct params {
    std::string model;
    std::string mmproj;
    std::string lora;
    std::string devices;
    std::string image;
    std::string prompt;
    std::string retrieval_query;
    std::string pd_prefill_out;
    std::string pd_decode_in;
    std::string input_jsonl;
    std::string output_jsonl;
    int32_t     n_ctx          = 16384;
    int32_t     n_gpu_layers   = 999;
    int32_t     n_threads      = 0;
    uint64_t    cpu_mask       = 0;
    uint32_t    cpu_poll       = 50;
    int32_t     max_iterations = 256;
    int32_t     mask_token_id  = LLAMA_TOKEN_NULL;
    int32_t     exact_image_max_edge       = 0;
    lladao::d2f_prefix_prefill_mode prefix_prefill_mode = lladao::d2f_prefix_prefill_mode::exact;
    int32_t     prefix_pack_size = 512;
    int32_t     vision_kv_tile_size = 16;
    int32_t     vision_kv_topk_tiles = 20;
    int32_t     vision_kv_query_window = 32;
    int32_t     vision_kv_score_layers = 4;
    int32_t     vision_kv_pool_kernel = 7;
    float       vision_kv_keep_ratio = 0.75f;
    ggml_type   cache_type_k   = GGML_TYPE_F16;
    ggml_type   cache_type_v   = GGML_TYPE_F16;
    lladao::d2f_flash_attention_mode flash_attention_mode =
            lladao::d2f_flash_attention_mode::auto_select;
    int32_t     full_page_tile_size       = 980;
    int32_t     tile_retrieval_topk       = 0;
    int32_t     tile_retrieval_mask_rounds = 2;
    int32_t     yarn_orig_ctx             = 16384;
    float       yarn_factor               = 1.0f;
    int8_t      vision_gpu     = -1;
    bool        full_page_tiles = false;
    bool        full_page_overview = true;
    bool        exact_image = false;
    bool        preprocess_only = false;
    bool        prefix_cache = true;
    bool        generation_block_cache = false;
    bool        sparse_generation_logits = false;
    bool        cpu_strict = false;
    bool        release_vision_after_encode = false;
    bool        vision_kv_compression = false;
    bool        use_mmap = true;
    bool        print_timings = true;
};

void print_usage(const char * program) {
    std::fprintf(stderr,
            "Usage: %s --model MODEL.gguf --mmproj MMPROJ.gguf --image IMAGE "
            "--prompt GUI_INSTRUCTION [options]\n"
            "       %s --model MODEL.gguf --pd-decode-in PREFIX.state [options]\n"
            "       %s --model MODEL.gguf --mmproj MMPROJ.gguf --input-jsonl INPUT.jsonl "
            "--output-jsonl OUTPUT.jsonl [options]\n"
            "\n"
            "Required:\n"
            "  --model PATH          LLaDA-o language GGUF\n"
            "  --mmproj PATH         LLaDA-o vision projector GGUF (P and normal modes)\n"
            "  --image PATH          Screenshot to ground (P and normal modes)\n"
            "  --prompt TEXT         Exact GUI instruction (P and normal modes)\n"
            "\n"
            "Options:\n"
            "  --lora PATH           Optional F32 D2F LoRA GGUF\n"
            "  --ctx-size N          Context capacity (default: 16384)\n"
            "  --gpu-layers N        Layers to offload (default: 999)\n"
            "  --device NAMES        Explicit offload devices separated by comma or slash\n"
            "  --no-mmap             Load model weights without mmap (recommended for Hexagon)\n"
            "  --vision-gpu          Offload the vision projector even with --gpu-layers 0\n"
            "  --no-vision-gpu       Keep the vision projector on the CPU\n"
            "  --threads N           CPU threads (default: backend choice)\n"
            "  --cpu-mask HEX        Bind language workers to this 64-bit CPU mask; 0 disables\n"
            "  --cpu-strict          Assign one selected CPU to each language worker\n"
            "  --cpu-poll N          Language worker polling level, 0..100 (default: 50)\n"
            "  --max-iterations N    D2F iteration limit (default: 256)\n"
            "  --cache-type-k TYPE   K cache type: f16 or q8_0 (default: f16)\n"
            "  --cache-type-v TYPE   V cache type: f16 or q8_0 (default: f16)\n"
            "  --flash-attn MODE     auto, enabled, or disabled (default: auto)\n"
            "  --no-prefix-cache     Recompute the complete sequence every D2F iteration\n"
            "  --generation-block-cache\n"
            "                        Reuse finalized leading generation blocks (default: disabled)\n"
            "  --no-generation-block-cache\n"
            "                        Disable finalized generation-block reuse\n"
            "  --sparse-generation-logits\n"
            "                        Request logits only for MASK rows (default: disabled)\n"
            "  --dense-generation-logits\n"
            "                        Request logits for every decoded generation row\n"
            "  --prefix-prefill-mode MODE\n"
            "                        exact, component_exact, component_parallel, packed_image,\n"
            "                        or packed_parallel\n"
            "                        (default: exact)\n"
            "  --prefix-pack-size N Image chunk or packed lane size (default: 512)\n"
            "  --vision-kv-compression\n"
            "                        Score and compact visual KV per layer and KV head\n"
            "  --vision-kv-tile-size N\n"
            "                        Patch tile edge used by KV scoring (default: 16)\n"
            "  --vision-kv-topk-tiles N\n"
            "                        Candidate visual tiles per image; 0 keeps all (default: 20)\n"
            "  --vision-kv-keep-ratio F\n"
            "                        Fraction of visual patches retained per head (default: 0.75)\n"
            "  --vision-kv-query-window N\n"
            "                        Last prompt queries used for scoring (default: 32)\n"
            "  --vision-kv-score-layers N\n"
            "                        Last language layers used for scoring (default: 4)\n"
            "  --vision-kv-pool-kernel N\n"
            "                        Odd 2D max-pool kernel for patch scores (default: 7)\n"
            "  --release-vision-after-encode\n"
            "                        Release the native vision context after caching embeddings\n"
            "  --pd-prefill-out PATH Prefill the exact prefix KV state, save it, and exit\n"
            "  --pd-decode-in PATH   Restore an exact prefix KV state and decode without vision input\n"
            "  --input-jsonl PATH    Run up to 100 validated requests through one engine\n"
            "  --output-jsonl PATH   Write and flush one result object per batch request\n"
            "  --mask-token-id N     Override missing tokenizer mask metadata\n"
            "  --full-page-tiles     Split the original page into exact row-major tiles\n"
            "  --full-page-tile-size N\n"
            "                        Tile edge in pixels (default: 980)\n"
            "  --no-full-page-overview\n"
            "                        Do not append the native-resized whole-page overview\n"
            "  --exact-image         Preserve image pixels and only pad to the patch grid\n"
            "  --exact-image-max-edge N\n"
            "                        Downscale a larger exact image before padding; 0 disables\n"
            "  --tile-retrieval-topk N\n"
            "                        Keep the Top-N source tiles; 0 keeps all (default: 0)\n"
            "  --retrieval-query TEXT\n"
            "                        Override the operation-only retrieval query\n"
            "  --tile-retrieval-mask-rounds N\n"
            "                        Complementary masked-query rounds (default: 2)\n"
            "  --yarn-factor F       YaRN RoPE factor; 1 disables scaling (default: 1)\n"
            "  --yarn-orig-ctx N     YaRN original context length (default: 16384)\n"
            "  --preprocess-only     Stop after validating multimodal layout\n"
            "  -h, --help            Show this help\n",
            program,
            program,
            program);
}

int32_t parse_i32(const char * flag, const char * value) {
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(std::string("invalid value for ") + flag + ": " + value);
    }
    return static_cast<int32_t>(parsed);
}

float parse_float(const char * flag, const char * value) {
    errno = 0;
    char * end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)) {
        throw std::invalid_argument(std::string("invalid value for ") + flag + ": " + value);
    }
    return parsed;
}

bool is_supported_cache_type(ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0;
}

bool is_valid_flash_attention_mode(lladao::d2f_flash_attention_mode mode) {
    return mode == lladao::d2f_flash_attention_mode::auto_select ||
           mode == lladao::d2f_flash_attention_mode::disabled ||
           mode == lladao::d2f_flash_attention_mode::enabled;
}

const char * flash_attention_mode_name(lladao::d2f_flash_attention_mode mode) {
    switch (mode) {
        case lladao::d2f_flash_attention_mode::auto_select: return "auto";
        case lladao::d2f_flash_attention_mode::disabled:    return "disabled";
        case lladao::d2f_flash_attention_mode::enabled:     return "enabled";
    }
    return "invalid";
}

std::string language_backend_name(int32_t gpu_layers) {
    const enum ggml_backend_dev_type device_type = gpu_layers == 0
            ? GGML_BACKEND_DEVICE_TYPE_CPU
            : GGML_BACKEND_DEVICE_TYPE_GPU;
    ggml_backend_dev_t device = ggml_backend_dev_by_type(device_type);
    if (!device && device_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!device) {
        return "unavailable";
    }
    const ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
    const char * name = registry ? ggml_backend_reg_name(registry) : ggml_backend_dev_name(device);
    return name && name[0] != '\0' ? name : "unknown";
}

llama_flash_attn_type to_llama_flash_attention_mode(lladao::d2f_flash_attention_mode mode) {
    switch (mode) {
        case lladao::d2f_flash_attention_mode::auto_select:
            return LLAMA_FLASH_ATTN_TYPE_AUTO;
        case lladao::d2f_flash_attention_mode::disabled:
            return LLAMA_FLASH_ATTN_TYPE_DISABLED;
        case lladao::d2f_flash_attention_mode::enabled:
            return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    throw std::invalid_argument("invalid flash attention mode");
}

ggml_type parse_cache_type(const char * flag, const char * value) {
    if (std::strcmp(value, "f16") == 0) {
        return GGML_TYPE_F16;
    }
    if (std::strcmp(value, "q8_0") == 0) {
        return GGML_TYPE_Q8_0;
    }
    throw std::invalid_argument(std::string(flag) + " must be f16 or q8_0: " + value);
}

lladao::d2f_prefix_prefill_mode parse_prefix_prefill_mode(const char * value) {
    if (std::strcmp(value, "exact") == 0) {
        return lladao::d2f_prefix_prefill_mode::exact;
    }
    if (std::strcmp(value, "component_exact") == 0) {
        return lladao::d2f_prefix_prefill_mode::component_exact;
    }
    if (std::strcmp(value, "component_parallel") == 0) {
        return lladao::d2f_prefix_prefill_mode::component_parallel;
    }
    if (std::strcmp(value, "packed_image") == 0) {
        return lladao::d2f_prefix_prefill_mode::packed_image;
    }
    if (std::strcmp(value, "packed_parallel") == 0) {
        return lladao::d2f_prefix_prefill_mode::packed_parallel;
    }
    throw std::invalid_argument(
            std::string("--prefix-prefill-mode must be exact, component_exact, component_parallel, packed_image, or packed_parallel: ") + value);
}

lladao::d2f_flash_attention_mode parse_flash_attention_mode(const char * value) {
    if (std::strcmp(value, "auto") == 0) {
        return lladao::d2f_flash_attention_mode::auto_select;
    }
    if (std::strcmp(value, "enabled") == 0) {
        return lladao::d2f_flash_attention_mode::enabled;
    }
    if (std::strcmp(value, "disabled") == 0) {
        return lladao::d2f_flash_attention_mode::disabled;
    }
    throw std::invalid_argument(
            std::string("--flash-attn must be auto, enabled, or disabled: ") + value);
}

params parse_args(int argc, char ** argv) {
    params result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[i];
        };

        if (arg == "--model" || arg == "-m") {
            result.model = next();
        } else if (arg == "--mmproj") {
            result.mmproj = next();
        } else if (arg == "--lora") {
            result.lora = next();
        } else if (arg == "--image") {
            result.image = next();
        } else if (arg == "--prompt" || arg == "-p") {
            result.prompt = next();
        } else if (arg == "--ctx-size") {
            result.n_ctx = parse_i32(arg.c_str(), next());
        } else if (arg == "--gpu-layers" || arg == "-ngl") {
            result.n_gpu_layers = parse_i32(arg.c_str(), next());
        } else if (arg == "--device" || arg == "-dev") {
            result.devices = next();
        } else if (arg == "--no-mmap") {
            result.use_mmap = false;
        } else if (arg == "--vision-gpu") {
            result.vision_gpu = 1;
        } else if (arg == "--no-vision-gpu") {
            result.vision_gpu = 0;
        } else if (arg == "--threads" || arg == "-t") {
            result.n_threads = parse_i32(arg.c_str(), next());
        } else if (arg == "--cpu-mask") {
            result.cpu_mask = lladao::detail::parse_cpu_mask_hex(next());
        } else if (arg == "--cpu-strict") {
            result.cpu_strict = true;
        } else if (arg == "--cpu-poll") {
            const int32_t poll = parse_i32(arg.c_str(), next());
            if (poll < 0 || poll > 100) {
                throw std::invalid_argument("--cpu-poll must be in [0, 100]");
            }
            result.cpu_poll = static_cast<uint32_t>(poll);
        } else if (arg == "--max-iterations") {
            result.max_iterations = parse_i32(arg.c_str(), next());
        } else if (arg == "--cache-type-k") {
            result.cache_type_k = parse_cache_type(arg.c_str(), next());
        } else if (arg == "--cache-type-v") {
            result.cache_type_v = parse_cache_type(arg.c_str(), next());
        } else if (arg == "--flash-attn") {
            result.flash_attention_mode = parse_flash_attention_mode(next());
        } else if (arg == "--no-prefix-cache") {
            result.prefix_cache = false;
        } else if (arg == "--generation-block-cache") {
            result.generation_block_cache = true;
        } else if (arg == "--no-generation-block-cache") {
            result.generation_block_cache = false;
        } else if (arg == "--sparse-generation-logits") {
            result.sparse_generation_logits = true;
        } else if (arg == "--dense-generation-logits") {
            result.sparse_generation_logits = false;
        } else if (arg == "--prefix-prefill-mode") {
            result.prefix_prefill_mode = parse_prefix_prefill_mode(next());
        } else if (arg == "--prefix-pack-size") {
            result.prefix_pack_size = parse_i32(arg.c_str(), next());
        } else if (arg == "--vision-kv-compression") {
            result.vision_kv_compression = true;
        } else if (arg == "--vision-kv-tile-size") {
            result.vision_kv_tile_size = parse_i32(arg.c_str(), next());
        } else if (arg == "--vision-kv-topk-tiles") {
            result.vision_kv_topk_tiles = parse_i32(arg.c_str(), next());
        } else if (arg == "--vision-kv-keep-ratio") {
            result.vision_kv_keep_ratio = parse_float(arg.c_str(), next());
        } else if (arg == "--vision-kv-query-window") {
            result.vision_kv_query_window = parse_i32(arg.c_str(), next());
        } else if (arg == "--vision-kv-score-layers") {
            result.vision_kv_score_layers = parse_i32(arg.c_str(), next());
        } else if (arg == "--vision-kv-pool-kernel") {
            result.vision_kv_pool_kernel = parse_i32(arg.c_str(), next());
        } else if (arg == "--release-vision-after-encode") {
            result.release_vision_after_encode = true;
        } else if (arg == "--pd-prefill-out") {
            result.pd_prefill_out = next();
        } else if (arg == "--pd-decode-in") {
            result.pd_decode_in = next();
        } else if (arg == "--input-jsonl") {
            result.input_jsonl = next();
        } else if (arg == "--output-jsonl") {
            result.output_jsonl = next();
        } else if (arg == "--mask-token-id") {
            result.mask_token_id = parse_i32(arg.c_str(), next());
        } else if (arg == "--full-page-tiles") {
            result.full_page_tiles = true;
        } else if (arg == "--full-page-tile-size") {
            result.full_page_tile_size = parse_i32(arg.c_str(), next());
        } else if (arg == "--no-full-page-overview") {
            result.full_page_overview = false;
        } else if (arg == "--exact-image") {
            result.exact_image = true;
        } else if (arg == "--exact-image-max-edge") {
            result.exact_image_max_edge = parse_i32(arg.c_str(), next());
        } else if (arg == "--tile-retrieval-topk") {
            result.tile_retrieval_topk = parse_i32(arg.c_str(), next());
            result.full_page_tiles = true;
        } else if (arg == "--retrieval-query") {
            result.retrieval_query = next();
        } else if (arg == "--tile-retrieval-mask-rounds") {
            result.tile_retrieval_mask_rounds = parse_i32(arg.c_str(), next());
        } else if (arg == "--yarn-factor") {
            result.yarn_factor = parse_float(arg.c_str(), next());
        } else if (arg == "--yarn-orig-ctx") {
            result.yarn_orig_ctx = parse_i32(arg.c_str(), next());
        } else if (arg == "--preprocess-only") {
            result.preprocess_only = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (result.model.empty()) {
        throw std::invalid_argument("--model is required");
    }
    if (result.input_jsonl.empty() != result.output_jsonl.empty()) {
        throw std::invalid_argument("--input-jsonl and --output-jsonl must be provided together");
    }
    const bool batch_mode = !result.input_jsonl.empty();
    if (batch_mode && (!result.image.empty() || !result.prompt.empty())) {
        throw std::invalid_argument("batch mode and --image/--prompt are mutually exclusive");
    }
    if (batch_mode && (!result.pd_prefill_out.empty() || !result.pd_decode_in.empty())) {
        throw std::invalid_argument("batch mode and PD separation are mutually exclusive");
    }
    if (!result.pd_prefill_out.empty() && !result.pd_decode_in.empty()) {
        throw std::invalid_argument("--pd-prefill-out and --pd-decode-in are mutually exclusive");
    }
    if (!batch_mode && result.pd_decode_in.empty() &&
        (result.mmproj.empty() || result.image.empty() || result.prompt.empty())) {
        throw std::invalid_argument(
                "--mmproj, --image, and --prompt are required outside PD decode mode");
    }
    if (batch_mode && result.mmproj.empty()) {
        throw std::invalid_argument("--mmproj is required in batch mode");
    }
    if ((!result.pd_prefill_out.empty() || !result.pd_decode_in.empty()) &&
        !result.prefix_cache) {
        throw std::invalid_argument("PD separation requires the exact prefix cache");
    }
    if ((!result.pd_prefill_out.empty() || !result.pd_decode_in.empty()) &&
        result.preprocess_only) {
        throw std::invalid_argument("--preprocess-only is not valid in PD mode");
    }
    if (result.n_ctx <= 0 || result.n_threads < 0 || result.max_iterations <= 0) {
        throw std::invalid_argument("context size, threads, and max iterations must be valid");
    }
    if (result.cpu_poll > 100) {
        throw std::invalid_argument("--cpu-poll must be in [0, 100]");
    }
    if (result.prefix_pack_size <= 0 || result.prefix_pack_size > result.n_ctx) {
        throw std::invalid_argument("--prefix-pack-size must be in [1, --ctx-size]");
    }
    if (result.prefix_prefill_mode != lladao::d2f_prefix_prefill_mode::exact && !result.prefix_cache) {
        throw std::invalid_argument("split prefix prefill modes require the prefix cache");
    }
    if (lladao::detail::prefix_prefill_requires_flash_attention(result.prefix_prefill_mode) &&
        result.flash_attention_mode == lladao::d2f_flash_attention_mode::disabled) {
        throw std::invalid_argument(
                std::string(lladao::detail::prefix_prefill_mode_name(result.prefix_prefill_mode)) +
                " requires Flash Attention");
    }
    if (result.vision_kv_compression &&
        result.prefix_prefill_mode != lladao::d2f_prefix_prefill_mode::component_parallel) {
        throw std::invalid_argument("--vision-kv-compression requires --prefix-prefill-mode component_parallel");
    }
    if (result.vision_kv_compression &&
        (!result.pd_prefill_out.empty() || !result.pd_decode_in.empty())) {
        throw std::invalid_argument("visual KV compression is not yet compatible with PD state files");
    }
    if (result.vision_kv_tile_size <= 0 || result.vision_kv_topk_tiles < 0 ||
        !(result.vision_kv_keep_ratio > 0.0f && result.vision_kv_keep_ratio <= 1.0f) ||
        result.vision_kv_query_window < 0 || result.vision_kv_score_layers <= 0 ||
        result.vision_kv_pool_kernel <= 0 || result.vision_kv_pool_kernel % 2 == 0) {
        throw std::invalid_argument("invalid visual KV compression parameters");
    }
    if (result.cache_type_v == GGML_TYPE_Q8_0 &&
        result.flash_attention_mode != lladao::d2f_flash_attention_mode::enabled) {
        throw std::invalid_argument("q8_0 V cache requires --flash-attn enabled");
    }
    if (result.full_page_tile_size <= 0 || result.full_page_tile_size > 980) {
        throw std::invalid_argument("--full-page-tile-size must be in [1, 980]");
    }
    if (result.tile_retrieval_topk < 0 || result.tile_retrieval_mask_rounds <= 0) {
        throw std::invalid_argument("retrieval Top-K and mask rounds must be valid");
    }
    if (result.exact_image && result.full_page_tiles) {
        throw std::invalid_argument("--exact-image and --full-page-tiles are mutually exclusive");
    }
    if (result.exact_image_max_edge < 0 || result.exact_image_max_edge > 980 ||
        (result.exact_image_max_edge > 0 &&
         (result.exact_image_max_edge < 14 || result.exact_image_max_edge % 14 != 0))) {
        throw std::invalid_argument("--exact-image-max-edge must be 0 or a multiple of 14 in [14, 980]");
    }
    if (result.exact_image_max_edge > 0 && !result.exact_image) {
        throw std::invalid_argument("--exact-image-max-edge requires --exact-image");
    }
    if (result.yarn_factor < 1.0f || result.yarn_orig_ctx <= 0) {
        throw std::invalid_argument("YaRN factor and original context must be valid");
    }
    return result;
}

using json = nlohmann::ordered_json;

struct batch_input {
    std::string id;
    std::string image;
    std::string prompt;
    std::string retrieval_query;
    std::string image_cache_key;
};

bool is_batch_input_field(const std::string & field) {
    return field == "id" || field == "image" || field == "prompt" ||
           field == "retrieval_query" || field == "image_cache_key";
}

std::string read_batch_string(
        const json & record,
        const char * field,
        size_t line_number,
        bool required) {
    const auto found = record.find(field);
    if (found == record.end()) {
        if (required) {
            throw std::invalid_argument(
                    "batch input line " + std::to_string(line_number) +
                    " is missing string field " + field);
        }
        return {};
    }
    if (!found->is_string()) {
        throw std::invalid_argument(
                "batch input line " + std::to_string(line_number) +
                " field " + field + " must be a string");
    }
    const std::string value = found->get<std::string>();
    if (required && value.empty()) {
        throw std::invalid_argument(
                "batch input line " + std::to_string(line_number) +
                " field " + field + " must not be empty");
    }
    return value;
}

std::vector<batch_input> read_batch_inputs(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open batch input: " + path);
    }

    std::vector<batch_input> result;
    std::unordered_map<std::string, size_t> id_lines;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            throw std::invalid_argument(
                    "batch input line " + std::to_string(line_number) + " must not be empty");
        }
        if (result.size() == 100) {
            throw std::invalid_argument("batch input must contain at most 100 requests");
        }

        json record;
        try {
            record = json::parse(line);
        } catch (const json::exception & error) {
            throw std::invalid_argument(
                    "invalid JSON on batch input line " + std::to_string(line_number) +
                    ": " + error.what());
        }
        if (!record.is_object()) {
            throw std::invalid_argument(
                    "batch input line " + std::to_string(line_number) + " must be a JSON object");
        }
        for (const auto & entry : record.items()) {
            if (!is_batch_input_field(entry.key())) {
                throw std::invalid_argument(
                        "batch input line " + std::to_string(line_number) +
                        " has unknown field " + entry.key());
            }
        }

        batch_input request;
        request.id = read_batch_string(record, "id", line_number, true);
        request.image = read_batch_string(record, "image", line_number, true);
        request.prompt = read_batch_string(record, "prompt", line_number, true);
        request.retrieval_query = read_batch_string(record, "retrieval_query", line_number, false);
        request.image_cache_key = read_batch_string(record, "image_cache_key", line_number, false);
        const auto inserted = id_lines.emplace(request.id, line_number);
        if (!inserted.second) {
            throw std::invalid_argument(
                    "duplicate batch id " + request.id + " on lines " +
                    std::to_string(inserted.first->second) + " and " +
                    std::to_string(line_number));
        }
        result.push_back(std::move(request));
    }
    if (input.bad()) {
        throw std::runtime_error("failed while reading batch input: " + path);
    }
    if (result.empty()) {
        throw std::invalid_argument("batch input must contain at least one request");
    }
    return result;
}

std::vector<llama_token> tokenize(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    int32_t count = llama_tokenize(
            vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    if (count == 0) {
        return {};
    }
    if (count > 0) {
        throw std::runtime_error("token count probe unexpectedly succeeded without output storage");
    }

    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    count = llama_tokenize(
            vocab,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_special,
            parse_special);
    if (count < 0) {
        throw std::runtime_error("tokenization failed for: " + text);
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

llama_token special_token(const llama_vocab * vocab, const char * text) {
    const std::vector<llama_token> tokens = tokenize(vocab, text, false, true);
    if (tokens.size() != 1 || std::strcmp(llama_vocab_get_text(vocab, tokens[0]), text) != 0) {
        throw std::runtime_error(std::string("model vocabulary is missing exact special token ") + text);
    }
    return tokens[0];
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t size = llama_token_to_piece(
            vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    if (size < 0) {
        buffer.resize(static_cast<size_t>(-size));
        size = llama_token_to_piece(
                vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    }
    if (size < 0) {
        throw std::runtime_error("failed to decode token " + std::to_string(token));
    }
    return std::string(buffer.data(), static_cast<size_t>(size));
}

class token_embedding_reader {
  public:
    token_embedding_reader(const std::string & path, int32_t expected_n_embd, int32_t expected_n_vocab) :
        path_(path) {
        gguf_init_params init_params = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ nullptr,
        };
        metadata_.reset(gguf_init_from_file(path.c_str(), init_params));
        if (!metadata_) {
            throw std::runtime_error("failed to read GGUF metadata: " + path);
        }

        tensor_id_ = gguf_find_tensor(metadata_.get(), "token_embd.weight");
        if (tensor_id_ < 0) {
            throw std::runtime_error("GGUF has no token_embd.weight tensor: " + path);
        }

        const int64_t * ne = gguf_get_tensor_ne(metadata_.get(), tensor_id_);
        n_embd_            = ne[0];
        n_vocab_           = ne[1];
        type_              = gguf_get_tensor_type(metadata_.get(), tensor_id_);
        row_bytes_         = ggml_row_size(type_, n_embd_);
        data_offset_       = gguf_get_data_offset(metadata_.get()) +
                       gguf_get_tensor_offset(metadata_.get(), tensor_id_);

        if (n_embd_ != expected_n_embd || n_vocab_ != expected_n_vocab) {
            throw std::runtime_error(
                    "token_embd.weight shape does not match the loaded model: GGUF [" +
                    std::to_string(n_embd_) + ", " + std::to_string(n_vocab_) + "], model [" +
                    std::to_string(expected_n_embd) + ", " + std::to_string(expected_n_vocab) + "]");
        }
        if (gguf_get_tensor_size(metadata_.get(), tensor_id_) !=
            row_bytes_ * static_cast<size_t>(n_vocab_)) {
            throw std::runtime_error("token_embd.weight is not stored as contiguous vocabulary rows");
        }

        traits_ = ggml_get_type_traits(type_);
        if (!traits_ || !traits_->to_float) {
            throw std::runtime_error(
                    "token_embd.weight type cannot be converted to F32: " +
                    std::string(ggml_type_name(type_)));
        }

        file_.open(path, std::ios::binary);
        if (!file_) {
            throw std::runtime_error("failed to open GGUF tensor data: " + path);
        }
        raw_.resize(row_bytes_);
    }

    const std::vector<float> & get(llama_token token) {
        if (token < 0 || token >= n_vocab_) {
            throw std::out_of_range("token ID outside token_embd.weight: " + std::to_string(token));
        }
        const auto found = cache_.find(token);
        if (found != cache_.end()) {
            return found->second;
        }

        const uint64_t offset =
                static_cast<uint64_t>(data_offset_) + static_cast<uint64_t>(token) * row_bytes_;
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        file_.read(reinterpret_cast<char *>(raw_.data()), static_cast<std::streamsize>(raw_.size()));
        if (!file_) {
            throw std::runtime_error(
                    "failed to read token embedding row " + std::to_string(token) + " from " + path_);
        }

        std::vector<float> embedding(static_cast<size_t>(n_embd_));
        traits_->to_float(raw_.data(), embedding.data(), n_embd_);
        return cache_.emplace(token, std::move(embedding)).first->second;
    }

  private:
    std::string path_;
    gguf_context_ptr metadata_;
    std::ifstream file_;
    int64_t tensor_id_ = -1;
    int64_t n_embd_    = 0;
    int64_t n_vocab_   = 0;
    ggml_type type_    = GGML_TYPE_COUNT;
    size_t row_bytes_  = 0;
    size_t data_offset_ = 0;
    const ggml_type_traits * traits_ = nullptr;
    std::vector<unsigned char> raw_;
    std::unordered_map<llama_token, std::vector<float>> cache_;
};

struct encoded_image {
    std::vector<float> embeddings;
    uint64_t embedding_copy_bytes = 0;
    double embedding_copy_seconds = 0.0;
    int32_t n_tokens = 0;
    int32_t grid_width = 0;
    int32_t grid_height = 0;
    int32_t source_index = 0;
    llama_pos dense_position = 0;
    bool overview = false;
};

using bitmap_ptr = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;

struct tile_box {
    int32_t x0;
    int32_t y0;
    int32_t x1;
    int32_t y1;
};

bitmap_ptr load_image(mtmd_context * mctx, const std::string & image_path) {
    mtmd_helper_bitmap_wrapper wrapper =
            mtmd_helper_bitmap_init_from_file(mctx, image_path.c_str(), false);
    bitmap_ptr bitmap(wrapper.bitmap, mtmd_bitmap_free);
    std::unique_ptr<mtmd_helper_video, decltype(&mtmd_helper_video_free)>
            video(wrapper.video_ctx, mtmd_helper_video_free);
    if (!bitmap) {
        throw std::runtime_error("failed to load screenshot: " + image_path);
    }
    if (mtmd_bitmap_is_audio(bitmap.get())) {
        throw std::runtime_error("LLaDA-o requires an image, not audio");
    }
    return bitmap;
}

encoded_image encode_bitmap(
        mtmd_context * mctx,
        const mtmd_bitmap * bitmap,
        int32_t n_embd,
        bool exact_image = false,
        int32_t exact_image_max_edge = 0) {
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (!chunks.ptr) {
        throw std::runtime_error("failed to allocate multimodal chunks");
    }
    const mtmd_bitmap * bitmaps[] = { bitmap };
    const std::string marker = mtmd_get_marker(mctx);
    mtmd_input_text input = {
        /*.text          =*/ marker.data(),
        /*.text_len      =*/ marker.size(),
        /*.add_special   =*/ false,
        /*.parse_special =*/ true,
    };
    const int32_t tokenize_result = mtmd_tokenize(mctx, chunks.ptr.get(), &input, bitmaps, 1);
    if (tokenize_result != 0) {
        throw std::runtime_error("mtmd failed to preprocess screenshot, code " +
                                 std::to_string(tokenize_result));
    }

    const mtmd_input_chunk * image_chunk = nullptr;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const mtmd_input_chunk * chunk = chunks[i];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            if (image_chunk) {
                throw std::runtime_error("one screenshot unexpectedly produced multiple image chunks");
            }
            image_chunk = chunk;
        }
    }
    if (!image_chunk) {
        throw std::runtime_error("mtmd produced no image chunk");
    }

    const size_t n_image_tokens = mtmd_input_chunk_get_n_tokens(image_chunk);
    if (n_image_tokens == 0 || n_image_tokens > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("invalid image token count");
    }
    if (mtmd_encode_chunk(mctx, image_chunk) != 0) {
        throw std::runtime_error("vision encoder failed");
    }
    const float * output = mtmd_get_output_embd(mctx);
    if (!output) {
        throw std::runtime_error("vision encoder returned no embeddings");
    }

    encoded_image result;
    result.n_tokens = static_cast<int32_t>(n_image_tokens);
    clip_image_size processed_size = {
        static_cast<int>(mtmd_bitmap_get_nx(bitmap)),
        static_cast<int>(mtmd_bitmap_get_ny(bitmap)),
    };
    if (exact_image) {
        if (exact_image_max_edge > 0 &&
            std::max(processed_size.width, processed_size.height) > exact_image_max_edge) {
            processed_size = mtmd_lladao_exact_image_size(processed_size, exact_image_max_edge);
        }
        processed_size = mtmd_lladao_exact_tile_size(processed_size);
    } else {
        processed_size = mtmd_lladao_image_size(processed_size);
    }
    result.grid_width = processed_size.width / 14;
    result.grid_height = processed_size.height / 14;
    if (result.grid_width <= 0 || result.grid_height <= 0 ||
        static_cast<int64_t>(result.grid_width) * result.grid_height != result.n_tokens) {
        throw std::runtime_error("vision patch grid does not match the encoded token count");
    }
    const size_t embedding_values =
            static_cast<size_t>(result.n_tokens) * static_cast<size_t>(n_embd);
    const auto embedding_copy_started = std::chrono::steady_clock::now();
    result.embeddings.assign(output, output + embedding_values);
    result.embedding_copy_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - embedding_copy_started).count();
    result.embedding_copy_bytes = static_cast<uint64_t>(embedding_values) * sizeof(float);
    return result;
}

encoded_image encode_image(
        mtmd_context * mctx,
        const std::string & image_path,
        int32_t n_embd,
        bool exact_image,
        int32_t exact_image_max_edge) {
    const bitmap_ptr bitmap = load_image(mctx, image_path);
    if (exact_image && exact_image_max_edge == 0 &&
        (mtmd_bitmap_get_nx(bitmap.get()) > 980 || mtmd_bitmap_get_ny(bitmap.get()) > 980)) {
        throw std::invalid_argument("exact image dimensions must not exceed 980 pixels");
    }
    return encode_bitmap(mctx, bitmap.get(), n_embd, exact_image, exact_image_max_edge);
}

std::vector<tile_box> full_page_tile_boxes(
        int32_t width,
        int32_t height,
        int32_t tile_size) {
    if (width <= 0 || height <= 0 || tile_size <= 0) {
        throw std::invalid_argument("full-page dimensions and tile size must be positive");
    }

    std::vector<tile_box> boxes;
    for (int32_t y = 0; y < height; y += tile_size) {
        for (int32_t x = 0; x < width; x += tile_size) {
            boxes.push_back({
                x,
                y,
                std::min(x + tile_size, width),
                std::min(y + tile_size, height),
            });
        }
    }
    return boxes;
}

bitmap_ptr crop_bitmap(const mtmd_bitmap * bitmap, const tile_box & box) {
    const int32_t source_width  = static_cast<int32_t>(mtmd_bitmap_get_nx(bitmap));
    const int32_t source_height = static_cast<int32_t>(mtmd_bitmap_get_ny(bitmap));
    if (box.x0 < 0 || box.y0 < 0 || box.x0 >= box.x1 || box.y0 >= box.y1 ||
        box.x1 > source_width || box.y1 > source_height) {
        throw std::invalid_argument("tile box lies outside the source image");
    }

    const int32_t width  = box.x1 - box.x0;
    const int32_t height = box.y1 - box.y0;
    const unsigned char * source = mtmd_bitmap_get_data(bitmap);
    if (!source) {
        throw std::runtime_error("source bitmap has no pixel data");
    }

    std::vector<unsigned char> pixels(
            static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    for (int32_t y = 0; y < height; ++y) {
        const size_t source_offset =
                (static_cast<size_t>(box.y0 + y) * source_width + box.x0) * 3;
        const size_t destination_offset = static_cast<size_t>(y) * width * 3;
        std::copy_n(
                source + source_offset,
                static_cast<size_t>(width) * 3,
                pixels.data() + destination_offset);
    }

    bitmap_ptr result(
            mtmd_bitmap_init(
                    static_cast<uint32_t>(width),
                    static_cast<uint32_t>(height),
                    pixels.data()),
            mtmd_bitmap_free);
    if (!result) {
        throw std::bad_alloc();
    }
    return result;
}

struct prefix {
    std::vector<float> embeddings;
    std::vector<llama_pos> positions;
    std::vector<int32_t> image_lengths;
    int32_t image_length = 0;
    int32_t total_length = 0;
    llama_pos generation_position = 0;
};

struct cached_prefix_layout {
    int32_t image_length = 0;
    int32_t total_length = 0;
    llama_pos prompt_position = 0;
    llama_pos generation_position = 0;
};

struct pd_artifact_metadata {
    cached_prefix_layout layout;
    llama_token mask = LLAMA_TOKEN_NULL;
    llama_token bos = LLAMA_TOKEN_NULL;
    llama_token eos = LLAMA_TOKEN_NULL;
    int32_t n_vocab = 0;
    int32_t n_embd = 0;
    int32_t yarn_orig_ctx = 0;
    float yarn_factor = 1.0f;
    uint64_t model_size = 0;
    uint64_t model_parameters = 0;
    bool has_adapter = false;
    float adapter_scale = 1.0f;
    uint64_t adapter_size = 0;
    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;
    lladao::d2f_prefix_prefill_mode prefix_prefill_mode =
            lladao::d2f_prefix_prefill_mode::exact;
    int32_t prefix_pack_size = 512;
    lladao::d2f_flash_attention_mode flash_attention_mode =
            lladao::d2f_flash_attention_mode::auto_select;
};

constexpr uint32_t D2F_PD_MAGIC = 0x32445044U;
constexpr int32_t D2F_PD_VERSION = 3;
constexpr size_t D2F_PD_METADATA_TOKENS = 28;

llama_token bits_to_token(uint32_t bits) {
    llama_token token;
    static_assert(sizeof(token) == sizeof(bits), "llama_token must store 32-bit metadata");
    std::memcpy(&token, &bits, sizeof(token));
    return token;
}

uint32_t token_to_bits(llama_token token) {
    uint32_t bits;
    std::memcpy(&bits, &token, sizeof(bits));
    return bits;
}

llama_token float_to_token(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits_to_token(bits);
}

float token_to_float(llama_token token) {
    const uint32_t bits = token_to_bits(token);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void append_u64(std::vector<llama_token> & tokens, uint64_t value) {
    tokens.push_back(bits_to_token(static_cast<uint32_t>(value)));
    tokens.push_back(bits_to_token(static_cast<uint32_t>(value >> 32)));
}

uint64_t read_u64(const std::vector<llama_token> & tokens, size_t index) {
    return static_cast<uint64_t>(token_to_bits(tokens.at(index))) |
           (static_cast<uint64_t>(token_to_bits(tokens.at(index + 1))) << 32);
}

uint64_t file_size(const std::string & path) {
    if (path.empty()) {
        return 0;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("failed to inspect file: " + path);
    }
    const std::streampos end = file.tellg();
    if (end < 0) {
        throw std::runtime_error("failed to determine file size: " + path);
    }
    return static_cast<uint64_t>(end);
}

std::vector<llama_token> encode_pd_metadata(const pd_artifact_metadata & metadata) {
    std::vector<llama_token> tokens;
    tokens.reserve(D2F_PD_METADATA_TOKENS);
    tokens.push_back(bits_to_token(D2F_PD_MAGIC));
    tokens.push_back(D2F_PD_VERSION);
    tokens.push_back(metadata.layout.image_length);
    tokens.push_back(metadata.layout.total_length);
    tokens.push_back(metadata.layout.prompt_position);
    tokens.push_back(metadata.layout.generation_position);
    tokens.push_back(D2F_GENERATION_LENGTH);
    tokens.push_back(D2F_BLOCK_LENGTH);
    tokens.push_back(metadata.mask);
    tokens.push_back(metadata.bos);
    tokens.push_back(metadata.eos);
    tokens.push_back(metadata.n_vocab);
    tokens.push_back(metadata.n_embd);
    tokens.push_back(metadata.yarn_orig_ctx);
    tokens.push_back(float_to_token(metadata.yarn_factor));
    append_u64(tokens, metadata.model_size);
    append_u64(tokens, metadata.model_parameters);
    tokens.push_back(metadata.has_adapter ? 1 : 0);
    tokens.push_back(float_to_token(metadata.adapter_scale));
    append_u64(tokens, metadata.adapter_size);
    tokens.push_back(static_cast<llama_token>(metadata.cache_type_k));
    tokens.push_back(static_cast<llama_token>(metadata.cache_type_v));
    tokens.push_back(static_cast<llama_token>(metadata.prefix_prefill_mode));
    tokens.push_back(metadata.prefix_pack_size);
    tokens.push_back(static_cast<llama_token>(metadata.flash_attention_mode));
    if (tokens.size() != D2F_PD_METADATA_TOKENS) {
        throw std::logic_error("PD metadata layout is inconsistent");
    }
    return tokens;
}

pd_artifact_metadata decode_pd_metadata(const std::vector<llama_token> & tokens) {
    if (tokens.size() != D2F_PD_METADATA_TOKENS || token_to_bits(tokens[0]) != D2F_PD_MAGIC) {
        throw std::runtime_error("file is not a LLaDA-o D2F PD state");
    }
    if (tokens[1] != D2F_PD_VERSION) {
        throw std::runtime_error("unsupported LLaDA-o D2F PD state version");
    }

    pd_artifact_metadata metadata;
    metadata.layout.image_length = tokens[2];
    metadata.layout.total_length = tokens[3];
    metadata.layout.prompt_position = tokens[4];
    metadata.layout.generation_position = tokens[5];
    if (tokens[6] != D2F_GENERATION_LENGTH || tokens[7] != D2F_BLOCK_LENGTH) {
        throw std::runtime_error("PD state uses incompatible D2F dimensions");
    }
    metadata.mask = tokens[8];
    metadata.bos = tokens[9];
    metadata.eos = tokens[10];
    metadata.n_vocab = tokens[11];
    metadata.n_embd = tokens[12];
    metadata.yarn_orig_ctx = tokens[13];
    metadata.yarn_factor = token_to_float(tokens[14]);
    metadata.model_size = read_u64(tokens, 15);
    metadata.model_parameters = read_u64(tokens, 17);
    metadata.has_adapter = tokens[19] != 0;
    metadata.adapter_scale = token_to_float(tokens[20]);
    metadata.adapter_size = read_u64(tokens, 21);
    metadata.cache_type_k = static_cast<ggml_type>(tokens[23]);
    metadata.cache_type_v = static_cast<ggml_type>(tokens[24]);
    metadata.prefix_prefill_mode = static_cast<lladao::d2f_prefix_prefill_mode>(tokens[25]);
    metadata.prefix_pack_size = tokens[26];
    metadata.flash_attention_mode = static_cast<lladao::d2f_flash_attention_mode>(tokens[27]);

    if (metadata.layout.image_length <= 0 ||
        metadata.layout.total_length < metadata.layout.image_length ||
        metadata.layout.prompt_position < 0 ||
        metadata.layout.generation_position < metadata.layout.prompt_position ||
        metadata.layout.total_length > std::numeric_limits<int32_t>::max() - D2F_GENERATION_LENGTH ||
        metadata.n_vocab <= 0 || metadata.n_embd <= 0 ||
        metadata.yarn_orig_ctx <= 0 || metadata.yarn_factor < 1.0f ||
        !is_supported_cache_type(metadata.cache_type_k) ||
        !is_supported_cache_type(metadata.cache_type_v) ||
        !lladao::detail::prefix_prefill_mode_valid(metadata.prefix_prefill_mode) ||
        metadata.prefix_pack_size <= 0 ||
        !is_valid_flash_attention_mode(metadata.flash_attention_mode) ||
        !std::isfinite(metadata.yarn_factor) || !std::isfinite(metadata.adapter_scale)) {
        throw std::runtime_error("PD state metadata is invalid");
    }
    return metadata;
}

pd_artifact_metadata peek_pd_metadata(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open PD state: " + path);
    }

    uint32_t state_magic = 0;
    uint32_t state_version = 0;
    uint32_t token_count = 0;
    file.read(reinterpret_cast<char *>(&state_magic), sizeof(state_magic));
    file.read(reinterpret_cast<char *>(&state_version), sizeof(state_version));
    file.read(reinterpret_cast<char *>(&token_count), sizeof(token_count));
    if (!file || state_magic == 0 || state_version == 0 ||
        token_count != D2F_PD_METADATA_TOKENS) {
        throw std::runtime_error("PD state header is invalid");
    }

    std::vector<llama_token> tokens(token_count);
    file.read(
            reinterpret_cast<char *>(tokens.data()),
            static_cast<std::streamsize>(tokens.size() * sizeof(tokens[0])));
    if (!file) {
        throw std::runtime_error("PD state metadata is truncated");
    }
    return decode_pd_metadata(tokens);
}

void append_embedding(std::vector<float> & destination, const std::vector<float> & source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

prefix build_prefix(
        token_embedding_reader & token_embeddings,
        const std::vector<const encoded_image *> & images,
        const std::vector<llama_token> & prompt_tokens,
        llama_token vision_start,
        llama_token vision_end,
        llama_pos prompt_position,
        int32_t n_embd) {
    prefix result;
    size_t image_tokens = 0;
    for (const encoded_image * image : images) {
        image_tokens += static_cast<size_t>(image->n_tokens) + 2;
    }
    const size_t prefix_tokens = image_tokens + prompt_tokens.size();
    result.embeddings.reserve(prefix_tokens * static_cast<size_t>(n_embd));
    result.positions.reserve(prefix_tokens);

    for (const encoded_image * image : images) {
        result.image_lengths.push_back(image->n_tokens + 2);
        append_embedding(result.embeddings, token_embeddings.get(vision_start));
        result.positions.push_back(image->dense_position);
        result.embeddings.insert(
                result.embeddings.end(), image->embeddings.begin(), image->embeddings.end());
        result.positions.insert(
                result.positions.end(),
                static_cast<size_t>(image->n_tokens),
                image->dense_position);
        append_embedding(result.embeddings, token_embeddings.get(vision_end));
        result.positions.push_back(image->dense_position);
        result.image_length += image->n_tokens + 2;
    }

    llama_pos position = prompt_position;
    for (llama_token token : prompt_tokens) {
        append_embedding(result.embeddings, token_embeddings.get(token));
        result.positions.push_back(position++);
    }
    result.total_length        = static_cast<int32_t>(result.positions.size());
    result.generation_position = position;
    return result;
}

lladao::detail::d2f_visual_prefix_layout build_visual_prefix_layout(
        const std::vector<const encoded_image *> & images,
        const prefix & input_prefix,
        llama_pos prompt_position) {
    lladao::detail::d2f_visual_prefix_layout result;
    int32_t token_start = 0;
    for (const encoded_image * image : images) {
        if (!image || image->n_tokens <= 0 || image->grid_width <= 0 || image->grid_height <= 0 ||
            static_cast<int64_t>(image->grid_width) * image->grid_height != image->n_tokens) {
            throw std::logic_error("invalid encoded image in visual KV prefix layout");
        }
        const int32_t token_end = token_start + image->n_tokens + 2;
        result.spans.push_back({
            token_start,
            token_end,
            token_start + 1,
            token_end - 1,
            image->grid_height,
            image->grid_width,
            image->dense_position,
        });
        token_start = token_end;
    }
    if (token_start != input_prefix.image_length) {
        throw std::logic_error("visual spans do not match the input prefix");
    }
    result.prompt_start = input_prefix.image_length;
    result.prefix_length = input_prefix.total_length;
    result.prompt_position = prompt_position;
    return result;
}

lladao::detail::d2f_visual_kv_keep_plan build_visual_kv_keep_plan_from_capture(
        const lladao::detail::d2f_post_rope_qk_capture & capture,
        const llama_model * model,
        const lladao::detail::d2f_visual_prefix_layout & layout,
        const lladao::detail::d2f_visual_kv_compression_config & config) {
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_head = llama_model_n_head(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    const int32_t n_embd = llama_model_n_embd_inp(model);
    if (n_layer <= 0 || n_head <= 0 || n_head_kv <= 0 || n_embd % n_head != 0) {
        throw std::logic_error("invalid model dimensions for visual KV scoring");
    }
    const int32_t head_dim = n_embd / n_head;
    const int32_t prompt_tokens = layout.prefix_length - layout.prompt_start;
    const std::vector<int32_t> layers = lladao::detail::select_visual_score_layers(
            n_layer,
            config.vision_score_layers,
            config.vision_score_layer_mode);
    std::vector<lladao::detail::d2f_visual_layer_scores> scores;
    scores.reserve(layers.size());

    for (int32_t layer : layers) {
        const auto * image_key = capture.find(
                lladao::detail::d2f_qk_capture_phase::image,
                lladao::detail::d2f_qk_tensor_kind::key,
                layer);
        const auto * prompt_query = capture.find(
                lladao::detail::d2f_qk_capture_phase::prompt,
                lladao::detail::d2f_qk_tensor_kind::query,
                layer);
        const auto * prompt_key = capture.find(
                lladao::detail::d2f_qk_capture_phase::prompt,
                lladao::detail::d2f_qk_tensor_kind::key,
                layer);
        if (!image_key || !prompt_query || !prompt_key ||
            image_key->shape.tokens != layout.prompt_start ||
            image_key->shape.heads != n_head_kv || image_key->shape.head_dim != head_dim ||
            prompt_query->shape.tokens != prompt_tokens ||
            prompt_query->shape.heads != n_head || prompt_query->shape.head_dim != head_dim ||
            prompt_key->shape.tokens != prompt_tokens ||
            prompt_key->shape.heads != n_head_kv || prompt_key->shape.head_dim != head_dim) {
            throw std::runtime_error("captured post-RoPE Q/K does not match the visual prefix layout");
        }

        std::vector<float> prefix_key;
        prefix_key.reserve(image_key->values.size() + prompt_key->values.size());
        prefix_key.insert(prefix_key.end(), image_key->values.begin(), image_key->values.end());
        prefix_key.insert(prefix_key.end(), prompt_key->values.begin(), prompt_key->values.end());
        scores.push_back(lladao::detail::compute_visual_attention_scores(
                {
                    layer,
                    prompt_tokens,
                    layout.prefix_length,
                    n_head,
                    n_head_kv,
                    head_dim,
                    1.0f / std::sqrt(static_cast<float>(head_dim)),
                    prompt_query->values.data(),
                    prefix_key.data(),
                },
                layout,
                config.vision_score_query_window));
    }

    return lladao::detail::build_visual_kv_keep_plan(
            n_layer,
            layout,
            std::move(scores),
            config);
}

class batch_owner {
  public:
    batch_owner(int32_t capacity, int32_t n_embd) : batch_(llama_batch_init(capacity, n_embd, 1)) {
        if (!batch_.embd || !batch_.pos || !batch_.n_seq_id || !batch_.seq_id || !batch_.logits) {
            throw std::bad_alloc();
        }
    }

    ~batch_owner() {
        llama_batch_free(batch_);
    }

    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;

    llama_batch & get() {
        return batch_;
    }

  private:
    llama_batch batch_;
};

void initialize_batch_rows(llama_batch & batch, int32_t n_tokens) {
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
}

double same_position_cross_entropy(
        const float * logits,
        int32_t n_vocab,
        llama_token target) {
    if (!logits || target < 0 || target >= n_vocab) {
        throw std::runtime_error("invalid logits row or masked-query target");
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        maximum = std::max(maximum, logits[token]);
    }

    double exponential_sum = 0.0;
    for (int32_t token = 0; token < n_vocab; ++token) {
        exponential_sum += std::exp(static_cast<double>(logits[token] - maximum));
    }
    return static_cast<double>(maximum) + std::log(exponential_sum) - logits[target];
}

struct masked_query_score {
    double value = 0.0;
    bool cancelled = false;
};

masked_query_score score_image_with_masked_query(
        llama_context * context,
        batch_owner & owned_batch,
        token_embedding_reader & token_embeddings,
        const encoded_image & image,
        const std::vector<llama_token> & clean_query,
        llama_token vision_start,
        llama_token vision_end,
        llama_token mask,
        llama_pos query_position,
        int32_t mask_rounds,
        int32_t n_embd,
        int32_t n_vocab,
        const std::atomic<bool> & cancel_requested) {
    if (clean_query.size() < 3) {
        throw std::invalid_argument("retrieval query must contain at least one non-boundary token");
    }

    const int32_t image_length = image.n_tokens + 2;
    const int32_t query_length = static_cast<int32_t>(clean_query.size());
    const int32_t document_length = image_length + query_length;
    const int32_t effective_rounds = std::min(mask_rounds, query_length - 2);
    double loss_sum = 0.0;
    int32_t scored_tokens = 0;

    for (int32_t round = 0; round < effective_rounds; ++round) {
        if (cancel_requested.load(std::memory_order_relaxed)) {
            masked_query_score result;
            result.cancelled = true;
            return result;
        }
        std::vector<int32_t> target_indices;
        for (int32_t query_index = 1 + round;
             query_index < query_length - 1;
             query_index += effective_rounds) {
            target_indices.push_back(query_index);
        }
        if (target_indices.empty()) {
            continue;
        }

        llama_batch & batch = owned_batch.get();
        initialize_batch_rows(batch, document_length);

        int32_t row = 0;
        auto append_token_embedding = [&](llama_token token, llama_pos position) {
            const std::vector<float> & embedding = token_embeddings.get(token);
            std::copy(
                    embedding.begin(),
                    embedding.end(),
                    batch.embd + static_cast<size_t>(row) * n_embd);
            batch.pos[row++] = position;
        };

        append_token_embedding(vision_start, image.dense_position);
        std::copy(
                image.embeddings.begin(),
                image.embeddings.end(),
                batch.embd + static_cast<size_t>(row) * n_embd);
        std::fill_n(batch.pos + row, image.n_tokens, image.dense_position);
        row += image.n_tokens;
        append_token_embedding(vision_end, image.dense_position);

        for (int32_t query_index = 0; query_index < query_length; ++query_index) {
            const bool is_target =
                    std::binary_search(target_indices.begin(), target_indices.end(), query_index);
            append_token_embedding(
                    is_target ? mask : clean_query[query_index],
                    query_position + query_index);
            if (is_target) {
                batch.logits[image_length + query_index] = 1;
            }
        }
        if (row != document_length) {
            throw std::logic_error("masked-query document layout is inconsistent");
        }

        llama_memory_clear(llama_get_memory(context), true);
        const int32_t decode_result = llama_decode(context, batch);
        if (decode_result != 0) {
            if (decode_result == 2 && cancel_requested.load(std::memory_order_relaxed)) {
                masked_query_score result;
                result.cancelled = true;
                return result;
            }
            throw std::runtime_error(
                    "masked-query llama_decode failed with code " +
                    std::to_string(decode_result));
        }

        for (int32_t query_index : target_indices) {
            const int32_t batch_index = image_length + query_index;
            const float * logits = llama_get_logits_ith(context, batch_index);
            loss_sum += same_position_cross_entropy(
                    logits, n_vocab, clean_query[query_index]);
            ++scored_tokens;
        }
    }

    const int32_t expected_tokens = query_length - 2;
    if (scored_tokens != expected_tokens) {
        throw std::logic_error(
                "complementary mask rounds scored " + std::to_string(scored_tokens) +
                " tokens, expected " + std::to_string(expected_tokens));
    }
    masked_query_score result;
    result.value = -loss_sum / static_cast<double>(scored_tokens);
    return result;
}

struct retrieval_result {
    std::vector<double> scores;
    std::vector<int32_t> selected_source_indices;
    double latency_seconds = 0.0;
    bool cancelled = false;
};

retrieval_result retrieve_image_tiles(
        llama_context * context,
        batch_owner & owned_batch,
        token_embedding_reader & token_embeddings,
        const std::vector<encoded_image> & source_images,
        const std::vector<llama_token> & clean_query,
        llama_token vision_start,
        llama_token vision_end,
        llama_token mask,
        llama_pos query_position,
        int32_t topk,
        int32_t mask_rounds,
        int32_t n_embd,
        int32_t n_vocab,
        const std::atomic<bool> & cancel_requested) {
    if (topk <= 0) {
        throw std::invalid_argument("tile retrieval Top-K must be positive");
    }

    const auto started = std::chrono::steady_clock::now();
    retrieval_result result;
    result.scores.reserve(source_images.size());
    for (const encoded_image & image : source_images) {
        const masked_query_score score = score_image_with_masked_query(
                context,
                owned_batch,
                token_embeddings,
                image,
                clean_query,
                vision_start,
                vision_end,
                mask,
                query_position,
                mask_rounds,
                n_embd,
                n_vocab,
                cancel_requested);
        if (score.cancelled) {
            result.cancelled = true;
            return result;
        }
        if (!std::isfinite(score.value)) {
            throw std::runtime_error("masked-query tile score is not finite");
        }
        result.scores.push_back(score.value);
        std::fprintf(
                stderr,
                "tile_retrieval score source=%" PRId32 " value=%.8f\n",
                image.source_index,
                score.value);
    }

    std::vector<int32_t> rank(source_images.size());
    std::iota(rank.begin(), rank.end(), 0);
    std::stable_sort(rank.begin(), rank.end(), [&](int32_t left, int32_t right) {
        if (result.scores[left] != result.scores[right]) {
            return result.scores[left] > result.scores[right];
        }
        return source_images[left].source_index < source_images[right].source_index;
    });
    rank.resize(std::min(static_cast<size_t>(topk), rank.size()));
    for (int32_t index : rank) {
        result.selected_source_indices.push_back(source_images[index].source_index);
    }
    std::sort(
            result.selected_source_indices.begin(),
            result.selected_source_indices.end());

    const auto finished = std::chrono::steady_clock::now();
    result.latency_seconds =
            std::chrono::duration<double>(finished - started).count();
    return result;
}

void fill_batch(
        llama_batch & batch,
        const prefix & input_prefix,
        const std::vector<int32_t> & generation_tokens,
        const lladao::detail::d2f_generation_decode_plan & plan,
        token_embedding_reader & token_embeddings,
        int32_t n_embd) {
    if (plan.first_position != 0) {
        throw std::logic_error("full-sequence D2F batch must start at generation position zero");
    }
    const int32_t n_tokens = input_prefix.total_length + plan.active_end;
    initialize_batch_rows(batch, n_tokens);

    std::copy(input_prefix.embeddings.begin(), input_prefix.embeddings.end(), batch.embd);
    std::copy(input_prefix.positions.begin(), input_prefix.positions.end(), batch.pos);

    for (int32_t i = 0; i < plan.active_end; ++i) {
        const std::vector<float> & embedding = token_embeddings.get(generation_tokens[i]);
        std::copy(
                embedding.begin(),
                embedding.end(),
                batch.embd + static_cast<size_t>(input_prefix.total_length + i) * n_embd);
        batch.pos[input_prefix.total_length + i] = input_prefix.generation_position + i;
    }

    for (int32_t position : plan.logit_positions) {
        batch.logits[input_prefix.total_length + position] = 1;
    }
}

void fill_prefix_batch_range(
        llama_batch & batch,
        const prefix & input_prefix,
        int32_t offset,
        int32_t length,
        int32_t n_embd) {
    if (offset < 0 || length <= 0 || offset > input_prefix.total_length - length) {
        throw std::invalid_argument("prefix chunk is outside the encoded prefix");
    }
    initialize_batch_rows(batch, length);
    const size_t embedding_offset = static_cast<size_t>(offset) * n_embd;
    const size_t embedding_count = static_cast<size_t>(length) * n_embd;
    std::copy_n(
            input_prefix.embeddings.data() + embedding_offset,
            embedding_count,
            batch.embd);
    std::copy_n(input_prefix.positions.data() + offset, length, batch.pos);
    batch.logits[length - 1] = 1;
}

struct packed_image_prefill_plan {
    int32_t streams = 0;
    int32_t lane_tokens = 0;
    int32_t padded_tokens = 0;
    int32_t padding_tokens = 0;
    std::vector<int32_t> lengths;
};

packed_image_prefill_plan make_packed_image_prefill_plan(
        const std::vector<int32_t> & image_lengths) {
    if (image_lengths.empty()) {
        throw std::invalid_argument("packed image prefill needs at least one span");
    }

    packed_image_prefill_plan result;
    result.lengths = image_lengths;
    result.streams = static_cast<int32_t>(image_lengths.size());
    const int32_t max_length = *std::max_element(image_lengths.begin(), image_lengths.end());
    if (max_length <= 0) {
        throw std::invalid_argument("packed image spans must not be empty");
    }

    result.lane_tokens = max_length;
    const int64_t padded_tokens = static_cast<int64_t>(result.streams)*result.lane_tokens;
    if (padded_tokens > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("packed image prefill exceeds int32 token capacity");
    }
    result.padded_tokens = static_cast<int32_t>(padded_tokens);

    int32_t actual_tokens = 0;
    for (int32_t stream = 0; stream < result.streams; ++stream) {
        const int32_t length = image_lengths[stream];
        if (length <= 0) {
            throw std::invalid_argument("packed image spans must not be empty");
        }
        actual_tokens += length;
    }
    result.padding_tokens = result.padded_tokens - actual_tokens;
    return result;
}

void fill_packed_image_prefill_batch(
        llama_batch & batch,
        const prefix & input_prefix,
        const packed_image_prefill_plan & plan,
        int32_t n_embd) {
    if (plan.streams != static_cast<int32_t>(plan.lengths.size()) ||
        plan.padded_tokens <= 0 || plan.lane_tokens <= 0) {
        throw std::invalid_argument("packed image prefill plan does not match the prefix");
    }

    initialize_batch_rows(batch, input_prefix.image_length);
    std::copy_n(
            input_prefix.embeddings.data(),
            static_cast<size_t>(input_prefix.image_length)*n_embd,
            batch.embd);
    std::copy_n(
            input_prefix.positions.data(),
            input_prefix.image_length,
            batch.pos);

    int32_t source_offset = 0;
    for (int32_t stream = 0; stream < plan.streams; ++stream) {
        const int32_t length = plan.lengths[stream];
        for (int32_t token = 0; token < length; ++token) {
            batch.seq_id[source_offset + token][0] = stream;
        }
        source_offset += length;
    }
    if (source_offset != input_prefix.image_length) {
        throw std::logic_error("packed image prefill did not consume the image prefix");
    }
    batch.logits[input_prefix.image_length - 1] = 1;
}

void fill_generation_batch(
        llama_batch & batch,
        llama_pos generation_position,
        const std::vector<int32_t> & generation_tokens,
        const lladao::detail::d2f_generation_decode_plan & plan,
        token_embedding_reader & token_embeddings,
        int32_t n_embd) {
    initialize_batch_rows(batch, plan.input_rows());
    for (int32_t position = plan.first_position; position < plan.active_end; ++position) {
        const int32_t row = position - plan.first_position;
        const std::vector<float> & embedding = token_embeddings.get(generation_tokens[position]);
        std::copy(
                embedding.begin(),
                embedding.end(),
                batch.embd + static_cast<size_t>(row) * n_embd);
        batch.pos[row] = generation_position + position;
        batch.logits[row] = plan.requests_logits(position) ? 1 : 0;
    }
}

std::vector<diffusion_d2f_candidate> collect_generation_candidates(
        llama_context * context,
        const lladao::detail::d2f_generation_decode_plan & plan,
        const std::vector<int32_t> & generation_tokens,
        llama_token mask_token_id,
        int32_t batch_prefix_rows,
        int32_t generation_length,
        int32_t n_vocab) {
    std::vector<diffusion_d2f_candidate> result(generation_length);
    for (int32_t position = plan.active_start; position < plan.active_end; ++position) {
        if (generation_tokens[position] != mask_token_id) {
            continue;
        }
        const int32_t batch_index = plan.batch_index(position, batch_prefix_rows);
        const float * logits = llama_get_logits_ith(context, batch_index);
        if (!logits) {
            throw std::runtime_error(
                    "language model returned no logits for D2F batch row " +
                    std::to_string(batch_index));
        }
        result[position] = diffusion_d2f_argmax_candidate(logits, n_vocab);
    }
    return result;
}

mtmd::context_ptr make_vision_context(
        const params & p,
        llama_model * model,
        bool exact_tile,
        int32_t exact_tile_max_edge) {
    const auto model_load_started = std::chrono::steady_clock::now();
    const lladao::detail::d2f_process_resource_snapshot model_load_resources_started =
            lladao::detail::snapshot_process_resources();
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = p.vision_gpu < 0 ? p.n_gpu_layers > 0 : p.vision_gpu != 0;
    mtmd_params.print_timings = p.print_timings;
    mtmd_params.lladao_exact_tile = exact_tile;
    mtmd_params.lladao_exact_tile_max_edge = exact_tile_max_edge;
    if (p.n_threads > 0) {
        mtmd_params.n_threads = p.n_threads;
    }

    mtmd::context_ptr context(
            mtmd_init_from_file(p.mmproj.c_str(), model, mtmd_params));
    if (!context) {
        throw std::runtime_error("failed to load LLaDA-o mmproj: " + p.mmproj);
    }
    if (!mtmd_support_vision(context.get())) {
        throw std::runtime_error("mmproj does not support vision");
    }
    const double model_load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - model_load_started).count();
    const lladao::d2f_phase_resource_usage model_load_resources =
            lladao::detail::resource_usage_delta(
                    model_load_resources_started,
                    lladao::detail::snapshot_process_resources());
    print_phase_resources(
            "model_load", model_load_seconds, model_load_resources, "vision");
    std::fprintf(
            stderr,
            "D2F model_load_config component=vision use_gpu=%s exact_tile=%s"
            " exact_tile_max_edge=%" PRId32 "\n",
            mtmd_params.use_gpu ? "true" : "false",
            exact_tile ? "true" : "false",
            exact_tile_max_edge);
    return context;
}

struct full_page_images {
    std::vector<encoded_image> sources;
    std::unique_ptr<encoded_image> overview;
    int32_t width = 0;
    int32_t height = 0;
    int32_t dense_image_length = 0;
};

full_page_images encode_full_page(
        const params & p,
        llama_model * model,
        int32_t n_embd) {
    mtmd::context_ptr native_context = make_vision_context(p, model, false, 0);
    bitmap_ptr page = load_image(native_context.get(), p.image);

    full_page_images result;
    result.width  = static_cast<int32_t>(mtmd_bitmap_get_nx(page.get()));
    result.height = static_cast<int32_t>(mtmd_bitmap_get_ny(page.get()));

    if (p.full_page_overview) {
        result.overview =
                std::make_unique<encoded_image>(
                        encode_bitmap(native_context.get(), page.get(), n_embd, false, 0));
        result.overview->overview = true;
    }
    native_context.reset();

    const std::vector<tile_box> boxes =
            full_page_tile_boxes(result.width, result.height, p.full_page_tile_size);
    mtmd::context_ptr exact_context = make_vision_context(p, model, true, 0);
    result.sources.reserve(boxes.size());
    llama_pos dense_position = 0;
    for (size_t index = 0; index < boxes.size(); ++index) {
        const bitmap_ptr tile = crop_bitmap(page.get(), boxes[index]);
        encoded_image image = encode_bitmap(exact_context.get(), tile.get(), n_embd, true, 0);
        image.source_index = static_cast<int32_t>(index);
        image.dense_position = dense_position;
        dense_position += image.n_tokens + 2;
        result.sources.push_back(std::move(image));
        std::fprintf(
                stderr,
                "full_page tile=%zu box=[%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32
                "] patches=%" PRId32 " dense_position=%" PRId32 "\n",
                index,
                boxes[index].x0,
                boxes[index].y0,
                boxes[index].x1,
                boxes[index].y1,
                result.sources.back().n_tokens,
                result.sources.back().dense_position);
    }

    if (result.overview) {
        result.overview->source_index = static_cast<int32_t>(result.sources.size());
        result.overview->dense_position = dense_position;
        dense_position += result.overview->n_tokens + 2;
    }
    result.dense_image_length = dense_position;
    return result;
}

std::string trim_text(const std::string & text) {
    auto is_space = [](unsigned char character) {
        return std::isspace(character) != 0;
    };
    const auto begin = std::find_if_not(text.begin(), text.end(), is_space);
    const auto end = std::find_if_not(text.rbegin(), text.rend(), is_space).base();
    return begin < end ? std::string(begin, end) : std::string();
}

std::string full_page_generation_prompt(
        const std::string & operation,
        int32_t tile_count,
        int32_t width,
        int32_t height,
        bool include_overview) {
    const std::string page_size =
            std::to_string(width) + "x" + std::to_string(height);
    if (include_overview) {
        return
            "The first " + std::to_string(tile_count) +
            " images are exact non-overlapping tiles from one " + page_size +
            " webpage screenshot, ordered left-to-right and then top-to-bottom. "
            "The final image is a resized overview of that same complete screenshot. "
            "Treat all images as one page and use the final overview as the global "
            "coordinate reference. " + operation +
            " Return the action and bounding box with coordinates normalized to the "
            "complete original screenshot in [0,1000].";
    }
    return
        "The following " + std::to_string(tile_count) +
        " images are non-overlapping tiles from one " + page_size +
        " webpage screenshot, ordered left-to-right and then top-to-bottom. "
        "Treat them as one complete page. " + operation +
        " Return the action and bounding box with coordinates normalized to the "
        "complete original screenshot in [0,1000].";
}

std::vector<llama_token> add_text_boundaries(
        const llama_vocab * vocab,
        const std::string & text) {
    std::vector<llama_token> tokens;
    tokens.push_back(llama_vocab_bos(vocab));
    const std::vector<llama_token> content = tokenize(vocab, text, false, false);
    tokens.insert(tokens.end(), content.begin(), content.end());
    tokens.push_back(llama_vocab_eos(vocab));
    return tokens;
}

int32_t selected_prefix_capacity(
        const std::vector<encoded_image> & sources,
        const encoded_image * overview,
        int32_t topk,
        size_t prompt_tokens) {
    std::vector<int32_t> source_lengths;
    source_lengths.reserve(sources.size());
    for (const encoded_image & image : sources) {
        source_lengths.push_back(image.n_tokens + 2);
    }
    std::sort(source_lengths.begin(), source_lengths.end(), std::greater<int32_t>());
    if (topk > 0 && static_cast<size_t>(topk) < source_lengths.size()) {
        source_lengths.resize(static_cast<size_t>(topk));
    }

    int64_t capacity = static_cast<int64_t>(prompt_tokens) + D2F_GENERATION_LENGTH;
    for (int32_t length : source_lengths) {
        capacity += length;
    }
    if (overview) {
        capacity += overview->n_tokens + 2;
    }
    if (capacity > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("selected full-page prefix is too large");
    }
    return static_cast<int32_t>(capacity);
}

void print_layout(
        const prefix & input_prefix,
        int32_t n_images,
        int32_t n_image_patches,
        int32_t dense_image_length,
        size_t n_prompt_tokens,
        int32_t n_ctx) {
    std::fprintf(stderr,
            "LLaDA-o input layout:\n"
            "  image split: %" PRId32 " selected spans, %" PRId32 " resident tokens "
            "(%" PRId32 " patches), dense layout %" PRId32 " tokens\n"
            "  text split:  %zu tokens (BOS + GUI instruction + EOS), "
            "RoPE positions %" PRId32 "..%" PRId32 "\n"
            "  D2F output:   %" PRId32 " masks, block length %" PRId32 ", RoPE positions %" PRId32 "..%" PRId32 "\n"
            "  active max:   %" PRId32 " tokens, configured context limit %" PRId32 "\n",
            n_images,
            input_prefix.image_length,
            n_image_patches,
            dense_image_length,
            n_prompt_tokens,
            input_prefix.generation_position - static_cast<llama_pos>(n_prompt_tokens),
            input_prefix.generation_position - 1,
            D2F_GENERATION_LENGTH,
            D2F_BLOCK_LENGTH,
            input_prefix.generation_position,
            input_prefix.generation_position + D2F_GENERATION_LENGTH - 1,
            input_prefix.total_length + D2F_GENERATION_LENGTH,
            n_ctx);
}

} // namespace

namespace lladao {

namespace detail {

uint64_t parse_cpu_mask_hex(const char * value) {
    if (value == nullptr) {
        throw std::invalid_argument("--cpu-mask must be a hexadecimal integer");
    }
    const std::string text(value);
    size_t offset = 0;
    if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        offset = 2;
    }
    if (offset == text.size()) {
        throw std::invalid_argument("--cpu-mask must be a hexadecimal integer: " + text);
    }

    uint64_t result = 0;
    for (size_t i = offset; i < text.size(); ++i) {
        const char c = text[i];
        uint64_t digit = 0;
        if (c >= '0' && c <= '9') {
            digit = static_cast<uint64_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            digit = static_cast<uint64_t>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            digit = static_cast<uint64_t>(c - 'A' + 10);
        } else {
            throw std::invalid_argument("--cpu-mask must be a hexadecimal integer: " + text);
        }
        if (result > (std::numeric_limits<uint64_t>::max() - digit) / 16) {
            throw std::invalid_argument("--cpu-mask exceeds 64 bits: " + text);
        }
        result = result * 16 + digit;
    }
    return result;
}

std::string format_cpu_mask(uint64_t mask) {
    char buffer[2 + 16 + 1];
    std::snprintf(buffer, sizeof(buffer), "0x%" PRIx64, mask);
    return buffer;
}

int32_t resolve_cpu_threads(uint64_t mask, int32_t requested_threads) {
    if (mask == 0 || requested_threads != 0) {
        return requested_threads;
    }
    int32_t selected = 0;
    while (mask != 0) {
        selected += static_cast<int32_t>(mask & 1U);
        mask >>= 1U;
    }
    return selected;
}

const char * language_device_mode(int32_t gpu_layers) {
    return gpu_layers == 0 ? "cpu_only" : "automatic";
}

} // namespace detail

struct d2f_engine::impl {
    params base;
    std::string backend = "not_initialized";
    int32_t cpu_threads_requested = 0;
    std::vector<ggml_backend_dev_t> language_devices;
    llama_model_ptr model;
    const llama_vocab * vocab = nullptr;
    int32_t n_vocab = 0;
    int32_t n_embd = 0;
    llama_token bos = LLAMA_TOKEN_NULL;
    llama_token eos = LLAMA_TOKEN_NULL;
    llama_token mask = LLAMA_TOKEN_NULL;
    llama_token vision_start = LLAMA_TOKEN_NULL;
    llama_token vision_end = LLAMA_TOKEN_NULL;
    std::unique_ptr<token_embedding_reader> token_embeddings;
    std::unique_ptr<lladao::detail::d2f_post_rope_qk_capture> qk_capture;
    mtmd::context_ptr native_vision;
    bool native_vision_exact = false;
    std::unique_ptr<encoded_image> cached_native_image;
    std::string cached_native_image_key;
    bool cached_native_image_exact = false;
    llama_adapter_lora_ptr adapter;
    std::string adapter_path;
    float adapter_scale = 1.0f;
    std::atomic<bool> cancel_requested { false };
    std::atomic<bool> active { false };
    llama_context_ptr context;
    ggml_threadpool_t cpu_threadpool = nullptr;
    decltype(ggml_threadpool_free) * cpu_threadpool_free_fn = nullptr;
    decltype(ggml_threadpool_pause) * cpu_threadpool_pause_fn = nullptr;
    std::unique_ptr<batch_owner> owned_batch;
    int32_t batch_capacity = 0;
    int32_t resident_capacity = 0;
    int32_t output_capacity = 0;
    int32_t sequence_capacity = 0;
    std::mutex operation_mutex;

    struct operation_affinity_guard {
        impl & owner;
#if defined(__linux__) || defined(__ANDROID__)
        cpu_set_t inherited_affinity {};
        bool inherited_affinity_valid = false;
#endif

        explicit operation_affinity_guard(impl & value) : owner(value) {
            if (owner.active_operation_affinity != nullptr) {
                throw std::logic_error("nested D2F affinity scope");
            }
            owner.active_operation_affinity = this;
        }

        void capture_before_language_decode() {
#if defined(__linux__) || defined(__ANDROID__)
            if (owner.base.cpu_mask == 0 || inherited_affinity_valid) {
                return;
            }
#if defined(__ANDROID__)
            if (sched_getaffinity(
                        0,
                        sizeof(inherited_affinity),
                        &inherited_affinity) != 0) {
                throw std::runtime_error("failed to capture caller CPU affinity before language decode");
            }
#else
            if (pthread_getaffinity_np(
                        pthread_self(),
                        sizeof(inherited_affinity),
                        &inherited_affinity) != 0) {
                throw std::runtime_error("failed to capture caller CPU affinity before language decode");
            }
#endif
            inherited_affinity_valid = true;
            std::fprintf(
                    stderr,
                    "D2F language affinity activated immediately before decode mask=%s\n",
                    lladao::detail::format_cpu_mask(owner.base.cpu_mask).c_str());
#endif
        }

        ~operation_affinity_guard() {
            owner.pause_cpu_threadpool();
#if defined(__linux__) || defined(__ANDROID__)
            if (inherited_affinity_valid &&
#if defined(__ANDROID__)
                sched_setaffinity(
                        0,
                        sizeof(inherited_affinity),
                        &inherited_affinity) != 0) {
#else
                pthread_setaffinity_np(
                        pthread_self(),
                        sizeof(inherited_affinity),
                        &inherited_affinity) != 0) {
#endif
                std::fprintf(stderr, "warning: failed to restore caller CPU affinity\n");
            }
#endif
            owner.active_operation_affinity = nullptr;
        }
    };
    operation_affinity_guard * active_operation_affinity = nullptr;

    explicit impl(const d2f_engine_params & p) {
        if (p.model_path.empty()) {
            throw std::invalid_argument("model path must not be empty");
        }
        if (p.context_size <= 0 || p.threads < 0 || p.max_iterations <= 0) {
            throw std::invalid_argument("context size, threads, and max iterations must be valid");
        }
        if (p.cpu_poll > 100) {
            throw std::invalid_argument("CPU threadpool polling level must be in [0, 100]");
        }
        const int32_t resolved_cpu_threads =
                lladao::detail::resolve_cpu_threads(p.cpu_mask, p.threads);
        if (p.cpu_mask != 0 &&
            (resolved_cpu_threads <= 0 || resolved_cpu_threads > GGML_MAX_N_THREADS)) {
            throw std::invalid_argument("CPU affinity needs a valid language worker count");
        }
        if (!lladao::detail::prefix_prefill_mode_valid(p.prefix_prefill_mode)) {
            throw std::invalid_argument("invalid prefix prefill mode");
        }
        if (p.prefix_pack_size <= 0 || p.prefix_pack_size > p.context_size) {
            throw std::invalid_argument("prefix pack size must be in [1, context size]");
        }
        if (p.prefix_prefill_mode != lladao::d2f_prefix_prefill_mode::exact && !p.prefix_cache) {
            throw std::invalid_argument("split prefix prefill modes require the prefix cache");
        }
        if (lladao::detail::prefix_prefill_requires_flash_attention(p.prefix_prefill_mode) &&
            p.flash_attention_mode == lladao::d2f_flash_attention_mode::disabled) {
            throw std::invalid_argument(
                    std::string(lladao::detail::prefix_prefill_mode_name(p.prefix_prefill_mode)) +
                    " prefill requires Flash Attention");
        }
        if (p.vision_kv_compression &&
            p.prefix_prefill_mode != lladao::d2f_prefix_prefill_mode::component_parallel) {
            throw std::invalid_argument("visual KV compression requires component_parallel prefill");
        }
        if (p.vision_kv_tile_size <= 0 || p.vision_kv_topk_tiles < 0 ||
            !(p.vision_kv_keep_ratio > 0.0f && p.vision_kv_keep_ratio <= 1.0f) ||
            p.vision_kv_query_window < 0 || p.vision_kv_score_layers <= 0 ||
            p.vision_kv_pool_kernel <= 0 || p.vision_kv_pool_kernel % 2 == 0) {
            throw std::invalid_argument("invalid visual KV compression parameters");
        }
        if (p.exact_image_max_edge < 0 || p.exact_image_max_edge > 980 ||
            (p.exact_image_max_edge > 0 &&
             (p.exact_image_max_edge < 14 || p.exact_image_max_edge % 14 != 0))) {
            throw std::invalid_argument("exact image max edge must be 0 or a multiple of 14 in [14, 980]");
        }
        if (!is_supported_cache_type(p.cache_type_k) || !is_supported_cache_type(p.cache_type_v)) {
            throw std::invalid_argument("D2F cache types must be f16 or q8_0");
        }
        if (!is_valid_flash_attention_mode(p.flash_attention_mode)) {
            throw std::invalid_argument("invalid Flash Attention mode");
        }
        if (p.cache_type_v == GGML_TYPE_Q8_0 &&
            p.flash_attention_mode != lladao::d2f_flash_attention_mode::enabled) {
            throw std::invalid_argument("q8_0 V cache requires Flash Attention enabled");
        }
        if (p.yarn_factor < 1.0f || p.yarn_original_context <= 0) {
            throw std::invalid_argument("YaRN factor and original context must be valid");
        }
        if (!std::isfinite(p.adapter_scale)) {
            throw std::invalid_argument("adapter scale must be finite");
        }

        base.model = p.model_path;
        base.mmproj = p.mmproj_path;
        base.devices = p.devices;
        base.n_ctx = p.context_size;
        base.n_gpu_layers = p.gpu_layers;
        cpu_threads_requested = p.threads;
        base.n_threads = resolved_cpu_threads;
        base.cpu_mask = p.cpu_mask;
        base.cpu_poll = p.cpu_poll;
        base.cpu_strict = p.cpu_strict;
        base.max_iterations = p.max_iterations;
        base.mask_token_id = p.mask_token_id;
        base.exact_image_max_edge = p.exact_image_max_edge;
        base.prefix_prefill_mode = p.prefix_prefill_mode;
        base.prefix_pack_size = p.prefix_pack_size;
        base.vision_kv_tile_size = p.vision_kv_tile_size;
        base.vision_kv_topk_tiles = p.vision_kv_topk_tiles;
        base.vision_kv_query_window = p.vision_kv_query_window;
        base.vision_kv_score_layers = p.vision_kv_score_layers;
        base.vision_kv_pool_kernel = p.vision_kv_pool_kernel;
        base.vision_kv_keep_ratio = p.vision_kv_keep_ratio;
        base.cache_type_k = p.cache_type_k;
        base.cache_type_v = p.cache_type_v;
        base.flash_attention_mode = p.flash_attention_mode;
        base.yarn_orig_ctx = p.yarn_original_context;
        base.yarn_factor = p.yarn_factor;
        base.vision_gpu = p.vision_gpu;
        base.prefix_cache = p.prefix_cache;
        base.generation_block_cache = p.generation_block_cache;
        base.sparse_generation_logits = p.sparse_generation_logits;
        base.release_vision_after_encode = p.release_vision_after_encode;
        base.vision_kv_compression = p.vision_kv_compression;
        base.use_mmap = p.use_mmap;
        base.print_timings = p.print_timings;

        const auto model_load_started = std::chrono::steady_clock::now();
        const lladao::detail::d2f_process_resource_snapshot model_load_resources_started =
                lladao::detail::snapshot_process_resources();
        ggml_backend_load_all();
        backend = language_backend_name(base.n_gpu_layers);

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = base.n_gpu_layers;
        model_params.load_mode = base.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        if (!base.devices.empty()) {
            size_t begin = 0;
            while (begin < base.devices.size()) {
                const size_t end = base.devices.find_first_of(",/", begin);
                std::string name = base.devices.substr(
                        begin, end == std::string::npos ? std::string::npos : end - begin);
                const size_t first = name.find_first_not_of(" \t");
                const size_t last = name.find_last_not_of(" \t");
                if (first == std::string::npos) {
                    throw std::invalid_argument("empty device in --device list");
                }
                name = name.substr(first, last - first + 1);
                ggml_backend_dev_t device = ggml_backend_dev_by_name(name.c_str());
                if (!device || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    throw std::invalid_argument("unknown non-CPU device: " + name);
                }
                language_devices.push_back(device);
                if (end == std::string::npos) {
                    break;
                }
                begin = end + 1;
            }
            language_devices.push_back(nullptr);
            model_params.devices = language_devices.data();
            backend = base.devices;
        } else if (base.n_gpu_layers == 0) {
            language_devices.push_back(nullptr);
            model_params.devices = language_devices.data();
        }
        model.reset(llama_model_load_from_file(base.model.c_str(), model_params));
        if (!model) {
            throw std::runtime_error("failed to load language model: " + base.model);
        }
        if (!llama_model_is_diffusion(model.get())) {
            throw std::runtime_error("language GGUF is not marked as a diffusion model");
        }

        vocab = llama_model_get_vocab(model.get());
        n_vocab = llama_vocab_n_tokens(vocab);
        n_embd = llama_model_n_embd_inp(model.get());
        if (n_vocab <= 0 || n_embd <= 0) {
            throw std::runtime_error("language model has invalid vocabulary or embedding width");
        }

        bos = llama_vocab_bos(vocab);
        eos = llama_vocab_eos(vocab);
        mask = llama_vocab_mask(vocab);
        if (mask == LLAMA_TOKEN_NULL) {
            mask = base.mask_token_id;
        }
        if (bos == LLAMA_TOKEN_NULL || eos == LLAMA_TOKEN_NULL || mask == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("model must provide BOS, EOS, and MASK tokens");
        }
        vision_start = special_token(vocab, "<|vision_start|>");
        vision_end = special_token(vocab, "<|vision_end|>");

        token_embeddings = std::make_unique<token_embedding_reader>(base.model, n_embd, n_vocab);

        if (base.vision_kv_compression) {
            const int32_t n_layer = llama_model_n_layer(model.get());
            const int32_t n_head = llama_model_n_head(model.get());
            const int32_t n_head_kv = llama_model_n_head_kv(model.get());
            if (n_layer <= 0 || n_head <= 0 || n_head_kv <= 0 || n_embd % n_head != 0 ||
                base.vision_kv_score_layers > n_layer) {
                throw std::runtime_error("language model does not satisfy the visual KV capture contract");
            }
            qk_capture = std::make_unique<lladao::detail::d2f_post_rope_qk_capture>(
                    lladao::detail::d2f_qk_capture_config {
                        true,
                        n_layer,
                        base.vision_kv_score_layers,
                        n_head,
                        n_head_kv,
                        n_embd / n_head,
                    });
        }

        if (!p.adapter_path.empty()) {
            set_adapter_locked(p.adapter_path, p.adapter_scale);
        }
        const double model_load_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - model_load_started).count();
        const lladao::d2f_phase_resource_usage model_load_resources =
                lladao::detail::resource_usage_delta(
                        model_load_resources_started,
                        lladao::detail::snapshot_process_resources());
        print_phase_resources(
                "model_load", model_load_seconds, model_load_resources, "language");
        std::fprintf(
                stderr,
                "D2F model_load_config component=language gpu_layers=%" PRId32
                " device_mode=%s adapter=%s backend=%s use_mmap=%s\n",
                base.n_gpu_layers,
                lladao::detail::language_device_mode(base.n_gpu_layers),
                p.adapter_path.empty() ? "false" : "true",
                backend.c_str(),
                base.use_mmap ? "true" : "false");
        std::fprintf(
                stderr,
                "D2F cache_type_k=%s cache_type_v=%s flash_attn_requested=%s"
                " prefix_prefill_mode=%s prefix_prefill_semantics=%s prefix_pack_size=%" PRId32
                " generation_block_cache=%s sparse_generation_logits=%s"
                " release_vision_after_encode=%s cpu_mask_requested=%s"
                " cpu_threads_requested=%" PRId32 " cpu_threads_configured=%" PRId32
                " cpu_strict_requested=%s cpu_poll_requested=%" PRIu32
                " language_device_mode=%s backend=%s vision_threadpool_shared=false\n",
                ggml_type_name(base.cache_type_k),
                ggml_type_name(base.cache_type_v),
                flash_attention_mode_name(base.flash_attention_mode),
                lladao::detail::prefix_prefill_mode_name(base.prefix_prefill_mode),
                lladao::detail::prefix_prefill_semantics_name(base.prefix_prefill_mode),
                base.prefix_pack_size,
                base.generation_block_cache ? "true" : "false",
                base.sparse_generation_logits ? "true" : "false",
                base.release_vision_after_encode ? "true" : "false",
                lladao::detail::format_cpu_mask(base.cpu_mask).c_str(),
                cpu_threads_requested,
                base.n_threads,
                base.cpu_strict ? "true" : "false",
                base.cpu_poll,
                lladao::detail::language_device_mode(base.n_gpu_layers),
                backend.c_str());
    }

    ~impl() {
        if (context && cpu_threadpool) {
            llama_detach_threadpool(context.get());
        }
        context.reset();
        if (cpu_threadpool && cpu_threadpool_free_fn) {
            cpu_threadpool_free_fn(cpu_threadpool);
            cpu_threadpool = nullptr;
        }
    }

    void apply_adapter() {
        if (!context) {
            return;
        }
        if (!adapter) {
            if (llama_set_adapters_lora(context.get(), nullptr, 0, nullptr) != 0) {
                throw std::runtime_error("failed to clear LoRA adapter");
            }
            return;
        }
        llama_adapter_lora * adapters[] = { adapter.get() };
        float scales[] = { adapter_scale };
        if (llama_set_adapters_lora(context.get(), adapters, 1, scales) != 0) {
            throw std::runtime_error("failed to apply LoRA adapter");
        }
    }

    void set_adapter_locked(const std::string & path, float scale) {
        if (path.empty()) {
            clear_adapter_locked();
            return;
        }
        if (!std::isfinite(scale)) {
            throw std::invalid_argument("adapter scale must be finite");
        }
        if (adapter && adapter_path == path && adapter_scale == scale) {
            return;
        }

        llama_adapter_lora_ptr replacement(llama_adapter_lora_init(model.get(), path.c_str()));
        if (!replacement) {
            throw std::runtime_error("failed to load LoRA: " + path);
        }
        if (context) {
            llama_adapter_lora * adapters[] = { replacement.get() };
            float scales[] = { scale };
            if (llama_set_adapters_lora(context.get(), adapters, 1, scales) != 0) {
                throw std::runtime_error("failed to apply LoRA adapter");
            }
        }
        adapter = std::move(replacement);
        adapter_path = path;
        adapter_scale = scale;
    }

    void clear_adapter_locked() {
        if (context && llama_set_adapters_lora(context.get(), nullptr, 0, nullptr) != 0) {
            throw std::runtime_error("failed to clear LoRA adapter");
        }
        adapter.reset();
        adapter_path.clear();
        adapter_scale = 1.0f;
    }

    void ensure_cpu_threadpool() {
        if (base.cpu_mask == 0 || cpu_threadpool) {
            return;
        }

        ggml_backend_dev_t cpu_device =
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_device) {
            throw std::runtime_error("CPU affinity requested but no CPU backend is available");
        }
        ggml_backend_reg_t cpu_registry =
                ggml_backend_dev_backend_reg(cpu_device);
        if (!cpu_registry) {
            throw std::runtime_error("CPU affinity requested but the CPU backend has no registry");
        }

        auto * new_fn = reinterpret_cast<decltype(ggml_threadpool_new) *>(
                ggml_backend_reg_get_proc_address(cpu_registry, "ggml_threadpool_new"));
        auto * free_fn = reinterpret_cast<decltype(ggml_threadpool_free) *>(
                ggml_backend_reg_get_proc_address(cpu_registry, "ggml_threadpool_free"));
        auto * pause_fn = reinterpret_cast<decltype(ggml_threadpool_pause) *>(
                ggml_backend_reg_get_proc_address(cpu_registry, "ggml_threadpool_pause"));
        if (!new_fn || !free_fn || !pause_fn) {
            throw std::runtime_error("CPU backend does not provide the ggml threadpool API");
        }

        ggml_threadpool_params threadpool_params =
                ggml_threadpool_params_default(base.n_threads);
        for (int32_t cpu = 0; cpu < 64; ++cpu) {
            threadpool_params.cpumask[cpu] =
                    (base.cpu_mask & (uint64_t(1) << cpu)) != 0;
        }
        threadpool_params.strict_cpu = base.cpu_strict;
        threadpool_params.poll = base.cpu_poll;
        threadpool_params.paused = true;

        ggml_threadpool_t threadpool = new_fn(&threadpool_params);
        if (!threadpool) {
            throw std::runtime_error(
                    "failed to create CPU threadpool with " +
                    std::to_string(base.n_threads) + " language workers");
        }

        cpu_threadpool_free_fn = free_fn;
        cpu_threadpool_pause_fn = pause_fn;
        cpu_threadpool = threadpool;
    }

    void pause_cpu_threadpool() noexcept {
        if (cpu_threadpool && cpu_threadpool_pause_fn) {
            cpu_threadpool_pause_fn(cpu_threadpool);
        }
    }

    void activate_language_affinity() {
        if (base.cpu_mask == 0) {
            return;
        }
        if (!active_operation_affinity) {
            throw std::logic_error("language affinity activated outside a D2F request");
        }
        active_operation_affinity->capture_before_language_decode();
    }

    int32_t resolved_cpu_threads() const {
        if (context) {
            return llama_n_threads(context.get());
        }
        return base.n_threads;
    }

    void ensure_context(
            int32_t required_resident,
            int32_t required_outputs,
            int32_t required_ubatch,
            int32_t required_sequences = 1) {
        if (required_resident <= 0 || required_outputs <= 0 || required_ubatch <= 0 ||
            required_ubatch > required_resident || required_sequences <= 0) {
            throw std::invalid_argument("invalid D2F context capacity request");
        }
        const lladao::detail::language_context_capacity capacity = {
            resident_capacity,
            batch_capacity,
            output_capacity,
            sequence_capacity,
        };
        if (!lladao::detail::language_context_needs_rebuild(
                    context != nullptr,
                    capacity,
                    required_resident,
                    required_ubatch,
                    required_outputs,
                    required_sequences)) {
            return;
        }

        owned_batch.reset();
        if (context && cpu_threadpool) {
            llama_detach_threadpool(context.get());
        }
        context.reset();

        llama_context_params context_params = llama_context_default_params();
        // The engine-level context size is an upper bound. Allocating a KV cache for
        // that entire limit (16K by default) wastes several GiB on mobile when the
        // current request contains only a few thousand resident tokens. The cache
        // stores one entry per submitted token, so size it to this request instead;
        // sparse multimodal RoPE positions do not require empty cache entries.
        context_params.n_ctx = required_resident;
        context_params.n_batch = required_ubatch;
        context_params.n_ubatch = required_ubatch;
        context_params.n_outputs_max = required_outputs;
        context_params.n_seq_max = required_sequences;
        context_params.kv_unified = true;
        context_params.no_perf = false;
        context_params.type_k = base.cache_type_k;
        context_params.type_v = base.cache_type_v;
        const bool require_parallel_flash =
                lladao::detail::prefix_prefill_requires_flash_attention(base.prefix_prefill_mode);
        context_params.flash_attn_type = require_parallel_flash
                ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                : to_llama_flash_attention_mode(base.flash_attention_mode);
        const bool language_gpu = base.n_gpu_layers != 0;
        context_params.offload_kqv = language_gpu;
        context_params.op_offload = language_gpu;
        context_params.abort_callback = [](void * data) {
            return static_cast<std::atomic<bool> *>(data)->load(std::memory_order_relaxed);
        };
        context_params.abort_callback_data = &cancel_requested;
        if (qk_capture) {
            context_params.cb_eval = lladao::detail::d2f_post_rope_qk_capture::callback;
            context_params.cb_eval_user_data = qk_capture.get();
        }
        if (base.yarn_factor > 1.0f) {
            context_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
            context_params.rope_freq_scale = 1.0f / base.yarn_factor;
            context_params.yarn_orig_ctx = static_cast<uint32_t>(base.yarn_orig_ctx);
        }
        if (base.n_threads > 0) {
            context_params.n_threads = base.n_threads;
            context_params.n_threads_batch = base.n_threads;
        }

        context.reset(llama_init_from_model(model.get(), context_params));
        if (!context) {
            throw std::runtime_error("failed to create language context");
        }
        if (require_parallel_flash &&
            llama_context_flash_attn_type(context.get()) != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            context.reset();
            throw std::runtime_error(
                    std::string(lladao::detail::prefix_prefill_mode_name(base.prefix_prefill_mode)) +
                    " prefill did not resolve to Flash Attention");
        }
        ensure_cpu_threadpool();
        if (cpu_threadpool) {
            llama_attach_threadpool(context.get(), cpu_threadpool, nullptr);
        }
        std::fprintf(
                stderr,
                "D2F context resident_capacity=%" PRIu32 " request_tokens=%" PRId32
                " ubatch_capacity=%" PRIu32
                " sequence_capacity=%" PRIu32
                " configured_limit=%" PRId32 " language_gpu=%s cache_type_k=%s"
                " cache_type_v=%s flash_attn_requested=%s flash_attn_resolved=%s"
                " cpu_mask_resolved=%s cpu_threads_resolved=%" PRId32
                " cpu_strict_resolved=%s cpu_poll_resolved=%" PRIu32
                " cpu_threadpool_attached=%s language_device_mode=%s"
                " backend=%s vision_threadpool_shared=false\n",
                llama_n_ctx(context.get()),
                required_resident,
                llama_n_ubatch(context.get()),
                llama_n_seq_max(context.get()),
                base.n_ctx,
                language_gpu ? "true" : "false",
                ggml_type_name(base.cache_type_k),
                ggml_type_name(base.cache_type_v),
                flash_attention_mode_name(base.flash_attention_mode),
                llama_flash_attn_type_name(llama_context_flash_attn_type(context.get())),
                lladao::detail::format_cpu_mask(cpu_threadpool ? base.cpu_mask : 0).c_str(),
                resolved_cpu_threads(),
                cpu_threadpool && base.cpu_strict ? "true" : "false",
                cpu_threadpool ? base.cpu_poll : 0,
                cpu_threadpool ? "true" : "false",
                lladao::detail::language_device_mode(base.n_gpu_layers),
                backend.c_str());
        llama_set_causal_attn(context.get(), false);
        apply_adapter();

        owned_batch = std::make_unique<batch_owner>(required_ubatch, n_embd);
        batch_capacity = required_ubatch;
        resident_capacity = static_cast<int32_t>(llama_n_ctx(context.get()));
        output_capacity = required_outputs;
        sequence_capacity = required_sequences;
    }

    void set_cache_audit(d2f_result & result) const {
        result.language_device_mode =
                lladao::detail::language_device_mode(base.n_gpu_layers);
        result.cache_type_k = ggml_type_name(base.cache_type_k);
        result.cache_type_v = ggml_type_name(base.cache_type_v);
        result.flash_attention_requested = flash_attention_mode_name(base.flash_attention_mode);
        result.flash_attention_resolved = context
                ? llama_flash_attn_type_name(llama_context_flash_attn_type(context.get()))
                : "not_initialized";
        result.flash_attention_mode = result.flash_attention_resolved;
        result.prefix_prefill_mode =
                lladao::detail::prefix_prefill_mode_name(base.prefix_prefill_mode);
        result.prefix_prefill_semantics =
                lladao::detail::prefix_prefill_semantics_name(base.prefix_prefill_mode);
        result.backend = backend;
        result.prefix_pack_size = base.prefix_pack_size;
        result.vision_kv_compression = base.vision_kv_compression;
        result.generation_block_cache = base.prefix_cache && base.generation_block_cache;
        result.sparse_generation_logits = base.sparse_generation_logits;
        result.cpu_mask_requested = lladao::detail::format_cpu_mask(base.cpu_mask);
        result.cpu_mask_resolved =
                lladao::detail::format_cpu_mask(cpu_threadpool ? base.cpu_mask : 0);
        result.cpu_threads_requested = cpu_threads_requested;
        result.cpu_threads_resolved = resolved_cpu_threads();
        result.cpu_poll_requested = base.cpu_poll;
        result.cpu_poll_resolved = cpu_threadpool ? base.cpu_poll : 0;
        result.cpu_strict_requested = base.cpu_strict;
        result.cpu_strict_resolved = cpu_threadpool && base.cpu_strict;
        result.cpu_threadpool_attached = cpu_threadpool != nullptr;
        result.vision_threadpool_shared = false;
        result.release_vision_after_encode = base.release_vision_after_encode;
    }

    pd_artifact_metadata current_pd_metadata(const cached_prefix_layout & layout) const {
        pd_artifact_metadata metadata;
        metadata.layout = layout;
        metadata.mask = mask;
        metadata.bos = bos;
        metadata.eos = eos;
        metadata.n_vocab = n_vocab;
        metadata.n_embd = n_embd;
        metadata.yarn_orig_ctx = base.yarn_orig_ctx;
        metadata.yarn_factor = base.yarn_factor;
        metadata.model_size = llama_model_size(model.get());
        metadata.model_parameters = llama_model_n_params(model.get());
        metadata.has_adapter = adapter != nullptr;
        metadata.adapter_scale = adapter_scale;
        metadata.adapter_size = adapter ? file_size(adapter_path) : 0;
        metadata.cache_type_k = base.cache_type_k;
        metadata.cache_type_v = base.cache_type_v;
        metadata.prefix_prefill_mode = base.prefix_prefill_mode;
        metadata.prefix_pack_size = base.prefix_pack_size;
        metadata.flash_attention_mode = base.flash_attention_mode;
        return metadata;
    }

    void validate_pd_metadata(const pd_artifact_metadata & metadata) const {
        if (metadata.mask != mask || metadata.bos != bos || metadata.eos != eos ||
            metadata.n_vocab != n_vocab || metadata.n_embd != n_embd ||
            metadata.model_size != llama_model_size(model.get()) ||
            metadata.model_parameters != llama_model_n_params(model.get())) {
            throw std::runtime_error("PD state does not match the loaded language model");
        }
        if (metadata.yarn_orig_ctx != base.yarn_orig_ctx ||
            metadata.yarn_factor != base.yarn_factor) {
            throw std::runtime_error("PD state does not match the configured RoPE scaling");
        }
        if (metadata.cache_type_k != base.cache_type_k || metadata.cache_type_v != base.cache_type_v) {
            throw std::runtime_error("PD state does not match the configured KV cache types");
        }
        if (metadata.prefix_prefill_mode != base.prefix_prefill_mode ||
            metadata.prefix_pack_size != base.prefix_pack_size) {
            throw std::runtime_error("PD state does not match the configured prefix prefill mode");
        }
        if (metadata.flash_attention_mode != base.flash_attention_mode) {
            throw std::runtime_error("PD state does not match the configured Flash Attention mode");
        }
        if (metadata.has_adapter != (adapter != nullptr) ||
            metadata.adapter_scale != adapter_scale ||
            metadata.adapter_size != (adapter ? file_size(adapter_path) : 0)) {
            throw std::runtime_error("PD state does not match the loaded LoRA adapter");
        }
        const int32_t required_tokens =
                metadata.layout.total_length + D2F_GENERATION_LENGTH;
        if (required_tokens > base.n_ctx) {
            throw std::runtime_error(
                    "PD state needs " + std::to_string(required_tokens) +
                    " tokens but context size is " + std::to_string(base.n_ctx));
        }
    }

    uint64_t save_pd_state(
            const std::string & path,
            const cached_prefix_layout & layout) const {
        if (path.empty()) {
            throw std::invalid_argument("PD output path must not be empty");
        }
        const std::vector<llama_token> metadata =
                encode_pd_metadata(current_pd_metadata(layout));
        const size_t written = llama_state_seq_save_file(
                context.get(),
                path.c_str(),
                0,
                metadata.data(),
                metadata.size());
        if (written == 0) {
            throw std::runtime_error("failed to save PD state: " + path);
        }
        return written;
    }

    pd_artifact_metadata load_pd_state(
            const std::string & path,
            uint64_t & state_bytes) {
        const pd_artifact_metadata expected = peek_pd_metadata(path);
        validate_pd_metadata(expected);

        const int32_t required_tokens =
                expected.layout.total_length + D2F_GENERATION_LENGTH;
        ensure_context(
                required_tokens,
                D2F_GENERATION_LENGTH,
                D2F_GENERATION_LENGTH);
        llama_memory_clear(llama_get_memory(context.get()), true);

        std::vector<llama_token> metadata(D2F_PD_METADATA_TOKENS);
        size_t token_count = 0;
        const size_t read = llama_state_seq_load_file(
                context.get(),
                path.c_str(),
                0,
                metadata.data(),
                metadata.size(),
                &token_count);
        if (read == 0 || token_count != metadata.size()) {
            throw std::runtime_error("failed to restore PD state: " + path);
        }
        const pd_artifact_metadata restored = decode_pd_metadata(metadata);
        if (encode_pd_metadata(restored) != encode_pd_metadata(expected)) {
            throw std::runtime_error("PD state metadata changed while loading");
        }
        state_bytes = file_size(path);
        return restored;
    }

    d2f_result decode_prefix_locked(
            const cached_prefix_layout & layout,
            const prefix * input_prefix,
            bool prefix_cache,
            d2f_result result,
            std::chrono::steady_clock::time_point generation_started) {
        if (!prefix_cache && input_prefix == nullptr) {
            throw std::logic_error("full-sequence D2F decoding requires prefix embeddings");
        }
        const lladao::detail::d2f_process_resource_snapshot decode_resources_started =
                lladao::detail::snapshot_process_resources();

        activate_language_affinity();
        llama_set_d2f_attention(
                context.get(),
                layout.image_length,
                layout.total_length,
                layout.prompt_position,
                layout.generation_position,
                D2F_BLOCK_LENGTH);

        diffusion_d2f_scheduler_params scheduler_params;
        scheduler_params.generation_length = D2F_GENERATION_LENGTH;
        scheduler_params.block_length = D2F_BLOCK_LENGTH;
        scheduler_params.max_iterations = base.max_iterations;
        scheduler_params.mask_token_id = mask;
        scheduler_params.initial_token_id = bos;
        scheduler_params.eos_token_id = eos;
        diffusion_d2f_scheduler scheduler(scheduler_params);
        const bool generation_block_cache = prefix_cache && base.generation_block_cache;
        lladao::detail::d2f_decode_counters decode_counters;
        int32_t cached_complete_end = 0;
        result.generation_block_cache = generation_block_cache;
        result.sparse_generation_logits = base.sparse_generation_logits;

        const auto decode_started = std::chrono::steady_clock::now();
        while (!scheduler.done()) {
            if (cancel_requested.load(std::memory_order_relaxed)) {
                result.cancelled = true;
                break;
            }
            const diffusion_d2f_step step = scheduler.prepare_step();
            if (step.done) {
                break;
            }
            const auto iteration_started = std::chrono::steady_clock::now();
            const lladao::detail::d2f_generation_decode_plan plan =
                    lladao::detail::plan_d2f_generation_decode(
                            scheduler.tokens(),
                            mask,
                            step.active_start,
                            step.active_end,
                            D2F_BLOCK_LENGTH,
                            cached_complete_end,
                            generation_block_cache,
                            base.sparse_generation_logits);

            llama_batch & batch = owned_batch->get();
            if (prefix_cache) {
                if (!llama_memory_seq_rm(
                            llama_get_memory(context.get()),
                            0,
                            layout.generation_position + plan.first_position,
                            -1)) {
                    throw std::runtime_error("failed to discard the dynamic D2F cache range");
                }
                fill_generation_batch(
                        batch,
                        layout.generation_position,
                        scheduler.tokens(),
                        plan,
                        *token_embeddings,
                        n_embd);
            } else {
                llama_memory_clear(llama_get_memory(context.get()), true);
                fill_batch(
                        batch,
                        *input_prefix,
                        scheduler.tokens(),
                        plan,
                        *token_embeddings,
                        n_embd);
            }
            const int32_t decode_result = llama_decode(context.get(), batch);
            if (decode_result != 0) {
                if (decode_result == 2 && cancel_requested.load(std::memory_order_relaxed)) {
                    result.cancelled = true;
                    break;
                }
                throw std::runtime_error("llama_decode failed with code " + std::to_string(decode_result));
            }
            // AUTO flash attention is resolved by the backend for the live
            // context. Refresh after every successful decode so structured
            // results never report the requested mode as if it were resolved.
            set_cache_audit(result);

            const std::vector<diffusion_d2f_candidate> candidates =
                    collect_generation_candidates(
                            context.get(),
                            plan,
                            scheduler.tokens(),
                            mask,
                            prefix_cache ? 0 : layout.total_length,
                            D2F_GENERATION_LENGTH,
                            n_vocab);
            const std::vector<diffusion_d2f_update> updates = scheduler.apply_candidates(candidates);
            cached_complete_end = plan.next_cached_complete_end;
            decode_counters.add(plan, prefix_cache ? 0 : layout.total_length);
            const auto iteration_finished = std::chrono::steady_clock::now();
            ++result.iterations;
            std::fprintf(stderr,
                    "D2F iteration %" PRId32 ": active [%" PRId32 ", %" PRId32
                    "), decode [%" PRId32 ", %" PRId32 "), cached_complete=%" PRId32
                    ", blocks %" PRId32 ", updates %zu, decode_tokens=%" PRId32
                    ", rebuild_rows=%" PRId32 ", logit_rows=%zu"
                    ", seconds=%.6f\n",
                    step.iteration,
                    step.active_start,
                    step.active_end,
                    plan.first_position,
                    plan.active_end,
                    cached_complete_end,
                    step.blocks_added,
                    updates.size(),
                    batch.n_tokens,
                    plan.rebuild_rows(),
                    plan.logit_positions.size(),
                    std::chrono::duration<double>(iteration_finished - iteration_started).count());
        }

        result.d2f_input_rows = decode_counters.input_rows;
        result.d2f_active_rows = decode_counters.active_rows;
        result.d2f_rebuild_rows = decode_counters.rebuild_rows;
        result.d2f_logit_rows = decode_counters.logit_rows;
        result.d2f_reused_input_rows = decode_counters.reused_input_rows;

        const auto generation_finished = std::chrono::steady_clock::now();
        result.decode_seconds =
                std::chrono::duration<double>(generation_finished - decode_started).count();
        result.decode_resources = lladao::detail::resource_usage_delta(
                decode_resources_started,
                lladao::detail::snapshot_process_resources());
        result.generation_seconds =
                std::chrono::duration<double>(generation_finished - generation_started).count();
        std::fprintf(
                stderr,
                "D2F decode_seconds=%.6f generation_seconds=%.6f"
                " input_rows=%" PRIu64 " active_rows=%" PRIu64
                " rebuild_rows=%" PRIu64 " logit_rows=%" PRIu64
                " reused_input_rows=%" PRIu64 " generation_block_cache=%s"
                " sparse_generation_logits=%s\n",
                result.decode_seconds,
                result.generation_seconds,
                result.d2f_input_rows,
                result.d2f_active_rows,
                result.d2f_rebuild_rows,
                result.d2f_logit_rows,
                result.d2f_reused_input_rows,
                result.generation_block_cache ? "true" : "false",
                result.sparse_generation_logits ? "true" : "false");
        print_phase_resources("decode", result.decode_seconds, result.decode_resources);
        if (base.print_timings) {
            llama_perf_context_print(context.get());
        }
        if (result.cancelled) {
            return result;
        }

        const std::vector<int32_t> & generated = scheduler.tokens();
        const int32_t output_length = scheduler.output_length();
        for (int32_t i = 0; i < output_length; ++i) {
            if (generated[i] == mask) {
                throw std::runtime_error("D2F stopped with unresolved mask tokens");
            }
            result.text += token_piece(vocab, generated[i]);
        }
        return result;
    }

    d2f_result generate_locked(
            const d2f_request & request,
            const std::string & pd_state_out = {}) {
        params p = base;
        p.image = request.image_path;
        p.prompt = request.prompt;
        p.retrieval_query = request.retrieval_query;
        p.full_page_tile_size = request.full_page_tile_size;
        p.tile_retrieval_topk = request.tile_retrieval_topk;
        p.tile_retrieval_mask_rounds = request.tile_retrieval_mask_rounds;
        p.full_page_tiles = request.full_page_tiles || request.tile_retrieval_topk > 0;
        p.full_page_overview = request.full_page_overview;
        p.exact_image = request.exact_image;
        p.preprocess_only = request.preprocess_only;

        if (!pd_state_out.empty() && (!p.prefix_cache || p.preprocess_only)) {
            throw std::invalid_argument("PD prefill requires the exact prefix cache and language prefill");
        }
        if (p.image.empty() || p.prompt.empty()) {
            throw std::invalid_argument("image path and prompt must not be empty");
        }
        if (p.mmproj.empty()) {
            throw std::invalid_argument("mmproj path is required for image prefill");
        }
        if (p.full_page_tile_size <= 0 || p.full_page_tile_size > 980) {
            throw std::invalid_argument("full-page tile size must be in [1, 980]");
        }
        if (p.tile_retrieval_topk < 0 || p.tile_retrieval_mask_rounds <= 0) {
            throw std::invalid_argument("retrieval Top-K and mask rounds must be valid");
        }
        if (p.exact_image && p.full_page_tiles) {
            throw std::invalid_argument("exact image and full-page tiles are mutually exclusive");
        }

        const std::string operation = trim_text(p.prompt);
        const std::string retrieval_query =
                trim_text(p.retrieval_query.empty() ? operation : p.retrieval_query);
        if (operation.empty() || retrieval_query.empty()) {
            throw std::invalid_argument("GUI operation and retrieval query must not be blank");
        }

        d2f_result result;
        set_cache_audit(result);
        const auto vision_phase_started = std::chrono::steady_clock::now();
        const lladao::detail::d2f_process_resource_snapshot vision_resources_started =
                lladao::detail::snapshot_process_resources();
        std::vector<encoded_image> native_images;
        full_page_images page;
        std::vector<const encoded_image *> selected_images;
        std::vector<llama_token> prompt_tokens;
        llama_pos prompt_position = 1;
        int32_t dense_image_length = 0;
        int32_t max_work_tokens = 0;
        std::vector<llama_token> retrieval_query_tokens;
        uint64_t vision_embedding_copy_bytes = 0;
        double vision_embedding_copy_seconds = 0.0;

        if (p.full_page_tiles) {
            native_vision.reset();
            const auto vision_started = std::chrono::steady_clock::now();
            page = encode_full_page(p, model.get(), n_embd);
            result.vision_encode_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - vision_started).count();
            if (page.sources.empty()) {
                throw std::runtime_error("full-page tiling produced no source images");
            }
            for (const encoded_image & image : page.sources) {
                vision_embedding_copy_bytes += image.embedding_copy_bytes;
                vision_embedding_copy_seconds += image.embedding_copy_seconds;
            }
            if (page.overview) {
                vision_embedding_copy_bytes += page.overview->embedding_copy_bytes;
                vision_embedding_copy_seconds += page.overview->embedding_copy_seconds;
            }
            const std::string generation_prompt = full_page_generation_prompt(
                    operation,
                    static_cast<int32_t>(page.sources.size()),
                    page.width,
                    page.height,
                    page.overview != nullptr);
            prompt_tokens = add_text_boundaries(vocab, generation_prompt);
            prompt_position = page.dense_image_length;
            dense_image_length = page.dense_image_length;
            max_work_tokens = selected_prefix_capacity(
                    page.sources,
                    page.overview.get(),
                    p.tile_retrieval_topk,
                    prompt_tokens.size());

            if (p.tile_retrieval_topk > 0) {
                retrieval_query_tokens = add_text_boundaries(vocab, retrieval_query);
                int32_t maximum_score_document = 0;
                for (const encoded_image & source : page.sources) {
                    maximum_score_document = std::max(
                            maximum_score_document,
                            source.n_tokens + 2 +
                                    static_cast<int32_t>(retrieval_query_tokens.size()));
                }
                max_work_tokens = std::max(max_work_tokens, maximum_score_document);
            } else {
                for (const encoded_image & source : page.sources) {
                    selected_images.push_back(&source);
                }
            }
            if (page.overview && p.tile_retrieval_topk == 0) {
                selected_images.push_back(page.overview.get());
            }
        } else {
            prompt_tokens = add_text_boundaries(vocab, operation);
            if (!request.image_cache_key.empty() && cached_native_image &&
                cached_native_image_key == request.image_cache_key &&
                cached_native_image_exact == p.exact_image) {
                result.vision_cache_hit = true;
                selected_images.push_back(cached_native_image.get());
            } else {
                if (lladao::detail::vision_context_needs_rebuild(
                            native_vision != nullptr,
                            native_vision_exact,
                            p.exact_image)) {
                    native_vision.reset();
                    native_vision = make_vision_context(
                            base,
                            model.get(),
                            p.exact_image,
                            p.exact_image ? base.exact_image_max_edge : 0);
                    native_vision_exact = p.exact_image;
                }
                const auto vision_started = std::chrono::steady_clock::now();
                encoded_image image = encode_image(
                        native_vision.get(),
                        p.image,
                        n_embd,
                        p.exact_image,
                        base.exact_image_max_edge);
                result.vision_encode_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - vision_started).count();
                vision_embedding_copy_bytes = image.embedding_copy_bytes;
                vision_embedding_copy_seconds = image.embedding_copy_seconds;
                image.dense_position = 0;
                if (request.image_cache_key.empty()) {
                    native_images.push_back(std::move(image));
                    selected_images.push_back(&native_images.back());
                } else {
                    cached_native_image = std::make_unique<encoded_image>(std::move(image));
                    cached_native_image_key = request.image_cache_key;
                    cached_native_image_exact = p.exact_image;
                    selected_images.push_back(cached_native_image.get());
                }
            }
            const bool released_now = lladao::detail::release_vision_context(
                    native_vision,
                    base.release_vision_after_encode);
            result.vision_context_released =
                    base.release_vision_after_encode && native_vision == nullptr;
            if (released_now) {
                std::fprintf(
                        stderr,
                        "D2F released native vision context after image embedding encode\n");
            }
            dense_image_length = selected_images.back()->n_tokens + 2;
            max_work_tokens =
                    dense_image_length +
                    static_cast<int32_t>(prompt_tokens.size()) +
                    D2F_GENERATION_LENGTH;
            std::fprintf(
                    stderr,
                    "D2F vision_cache=%s key=%s encode_seconds=%.6f\n",
                    result.vision_cache_hit ? "hit" : "miss",
                    request.image_cache_key.empty() ? "none" : "content",
                    result.vision_encode_seconds);
        }

        result.vision_embedding_copy_bytes = vision_embedding_copy_bytes;
        result.vision_embedding_copy_seconds = vision_embedding_copy_seconds;
        result.vision_phase_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - vision_phase_started).count();
        result.vision_resources = lladao::detail::resource_usage_delta(
                vision_resources_started,
                lladao::detail::snapshot_process_resources());
        std::fprintf(
                stderr,
                "D2F vision_embedding_copy bytes=%" PRIu64 " seconds=%.6f\n",
                result.vision_embedding_copy_bytes,
                result.vision_embedding_copy_seconds);
        print_phase_resources("vision", result.vision_phase_seconds, result.vision_resources);

        if (max_work_tokens > p.n_ctx) {
            throw std::runtime_error(
                    "the largest scoring or generation batch needs " +
                    std::to_string(max_work_tokens) +
                    " tokens but context size is " + std::to_string(p.n_ctx));
        }

        if (p.preprocess_only) {
            if (selected_images.empty()) {
                const size_t count = std::min(
                        static_cast<size_t>(p.tile_retrieval_topk),
                        page.sources.size());
                for (size_t i = 0; i < count; ++i) {
                    selected_images.push_back(&page.sources[i]);
                }
                if (page.overview) {
                    selected_images.push_back(page.overview.get());
                }
                std::fprintf(
                        stderr,
                        "preprocess-only: retrieval scoring skipped; showing a %zu-tile capacity layout\n",
                        count);
            }
            const prefix input_prefix = build_prefix(
                    *token_embeddings,
                    selected_images,
                    prompt_tokens,
                    vision_start,
                    vision_end,
                    prompt_position,
                    n_embd);
            int32_t patch_count = 0;
            for (const encoded_image * image : selected_images) {
                patch_count += image->n_tokens;
            }
            print_layout(
                    input_prefix,
                    static_cast<int32_t>(selected_images.size()),
                    patch_count,
                    dense_image_length,
                    prompt_tokens.size(),
                    p.n_ctx);
            result.image_tokens = input_prefix.image_length;
            result.input_tokens = input_prefix.total_length;
            const int32_t prompt_length =
                    input_prefix.total_length - input_prefix.image_length;
            const std::vector<lladao::detail::prefix_chunk> prefix_chunks =
                    lladao::detail::plan_prefix_chunks(
                            input_prefix.image_lengths,
                            prompt_length,
                            base.prefix_prefill_mode,
                            base.prefix_pack_size);
            const lladao::detail::prefix_parallel_audit parallel_audit =
                    lladao::detail::audit_prefix_parallelism(
                            input_prefix.image_lengths,
                            base.prefix_prefill_mode,
                            false,
                            base.prefix_pack_size);
            result.prefix_chunk_count = static_cast<int32_t>(prefix_chunks.size());
            result.image_span_count = parallel_audit.domains;
            result.domains = parallel_audit.domains;
            result.attention_pairs_dense = static_cast<uint64_t>(
                    parallel_audit.attention_pairs_dense);
            result.attention_pairs_packed = static_cast<uint64_t>(
                    parallel_audit.attention_pairs_packed);
            result.preprocessed_only = true;
            return result;
        }

        const int32_t required_outputs = std::max<int32_t>(
                D2F_GENERATION_LENGTH,
                retrieval_query_tokens.empty()
                        ? 0
                        : static_cast<int32_t>(retrieval_query_tokens.size()));

        if (p.tile_retrieval_topk > 0) {
            ensure_context(max_work_tokens, required_outputs, max_work_tokens);
            set_cache_audit(result);
            llama_set_d2f_attention(context.get(), -1, -1, -1, -1, 0);
            activate_language_affinity();
            const retrieval_result retrieval = retrieve_image_tiles(
                    context.get(),
                    *owned_batch,
                    *token_embeddings,
                    page.sources,
                    retrieval_query_tokens,
                    vision_start,
                    vision_end,
                    mask,
                    prompt_position,
                    p.tile_retrieval_topk,
                    p.tile_retrieval_mask_rounds,
                    n_embd,
                    n_vocab,
                    cancel_requested);
            set_cache_audit(result);

            if (retrieval.cancelled) {
                result.cancelled = true;
                return result;
            }

            for (int32_t source_index : retrieval.selected_source_indices) {
                const auto found = std::find_if(
                        page.sources.begin(),
                        page.sources.end(),
                        [source_index](const encoded_image & image) {
                            return image.source_index == source_index;
                        });
                if (found == page.sources.end()) {
                    throw std::logic_error("retrieval selected an unknown source tile");
                }
                selected_images.push_back(&*found);
            }
            if (page.overview) {
                selected_images.push_back(page.overview.get());
            }

            std::fprintf(stderr, "tile_retrieval selected_sources=[");
            for (size_t i = 0; i < retrieval.selected_source_indices.size(); ++i) {
                std::fprintf(
                        stderr,
                        "%s%" PRId32,
                        i == 0 ? "" : ",",
                        retrieval.selected_source_indices[i]);
            }
            std::fprintf(
                    stderr,
                    "] overview=%s latency=%.6f query=%s\n",
                    page.overview ? "true" : "false",
                    retrieval.latency_seconds,
                    retrieval_query.c_str());
        }

        if (cancel_requested.load(std::memory_order_relaxed)) {
            result.cancelled = true;
            return result;
        }

        const auto prefill_phase_started = std::chrono::steady_clock::now();
        const lladao::detail::d2f_process_resource_snapshot prefill_resources_started =
                lladao::detail::snapshot_process_resources();
        const auto prefix_build_started = std::chrono::steady_clock::now();
        const prefix input_prefix = build_prefix(
                *token_embeddings,
                selected_images,
                prompt_tokens,
                vision_start,
                vision_end,
                prompt_position,
                n_embd);
        result.prefix_build_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - prefix_build_started).count();
        result.prefix_build_copy_bytes =
                static_cast<uint64_t>(input_prefix.embeddings.size()) * sizeof(float) +
                static_cast<uint64_t>(input_prefix.positions.size()) * sizeof(llama_pos);
        const int32_t max_tokens = input_prefix.total_length + D2F_GENERATION_LENGTH;
        if (max_tokens > p.n_ctx || max_tokens > max_work_tokens) {
            throw std::logic_error("selected prefix exceeds its validated capacity");
        }
        const int32_t prompt_length = input_prefix.total_length - input_prefix.image_length;
        const lladao::detail::d2f_visual_prefix_layout visual_prefix_layout =
                build_visual_prefix_layout(selected_images, input_prefix, prompt_position);
        const std::vector<lladao::detail::prefix_chunk> prefix_chunks =
                lladao::detail::plan_prefix_chunks(
                        input_prefix.image_lengths,
                        prompt_length,
                        base.prefix_prefill_mode,
                        base.prefix_pack_size);
        std::vector<int32_t> packed_lane_lengths;
        if (base.prefix_prefill_mode == lladao::d2f_prefix_prefill_mode::component_parallel &&
            input_prefix.image_lengths.size() > 1) {
            packed_lane_lengths = input_prefix.image_lengths;
        } else if (base.prefix_prefill_mode == lladao::d2f_prefix_prefill_mode::packed_parallel) {
            packed_lane_lengths = lladao::detail::plan_packed_parallel_lanes(
                    input_prefix.image_lengths,
                    base.prefix_pack_size);
        }
        const bool use_packed_parallel = packed_lane_lengths.size() > 1;
        const packed_image_prefill_plan packed_plan = use_packed_parallel
                ? make_packed_image_prefill_plan(packed_lane_lengths)
                : packed_image_prefill_plan {};
        const int32_t required_resident = max_work_tokens;
        int32_t required_ubatch = p.prefix_cache
                ? lladao::detail::prefix_chunk_ubatch(prefix_chunks, D2F_GENERATION_LENGTH)
                : max_tokens;
        result.prefix_chunk_count = static_cast<int32_t>(prefix_chunks.size());
        result.image_span_count = static_cast<int32_t>(selected_images.size());
        ensure_context(
                required_resident,
                required_outputs,
                required_ubatch,
                use_packed_parallel ? packed_plan.streams : 1);
        set_cache_audit(result);
        const bool flash_attention_resolved =
                llama_context_flash_attn_type(context.get()) == LLAMA_FLASH_ATTN_TYPE_ENABLED;
        lladao::detail::validate_prefix_parallel_backend(
                base.prefix_prefill_mode,
                flash_attention_resolved);
        const lladao::detail::prefix_parallel_audit parallel_audit =
                lladao::detail::audit_prefix_parallelism(
                        input_prefix.image_lengths,
                        base.prefix_prefill_mode,
                        flash_attention_resolved,
                        base.prefix_pack_size);
        result.domains = parallel_audit.domains;
        result.attention_pairs_dense = static_cast<uint64_t>(
                parallel_audit.attention_pairs_dense);
        result.attention_pairs_packed = static_cast<uint64_t>(
                parallel_audit.attention_pairs_packed);
        result.attention_pairs_executed = use_packed_parallel
                ? result.attention_pairs_packed
                : 0;
        result.attention_pairs_executed_known = use_packed_parallel;
        result.parallel_lane_tokens = use_packed_parallel ? packed_plan.lane_tokens : 0;
        result.parallel_padding_tokens = 0;
        int32_t selected_patches = 0;
        for (const encoded_image * image : selected_images) {
            selected_patches += image->n_tokens;
        }
        print_layout(
                input_prefix,
                static_cast<int32_t>(selected_images.size()),
                selected_patches,
                dense_image_length,
                prompt_tokens.size(),
                p.n_ctx);
        const int64_t observed_max_position =
                static_cast<int64_t>(input_prefix.generation_position) +
                D2F_GENERATION_LENGTH - 1;
        const int64_t declared_max_position = static_cast<int64_t>(
                static_cast<double>(p.yarn_orig_ctx) * p.yarn_factor);
        if (observed_max_position >= declared_max_position) {
            std::fprintf(
                    stderr,
                    "warning: maximum RoPE position %" PRId64
                    " reaches or exceeds the declared range [0, %" PRId64
                    "); this is unscaled or beyond-factor extrapolation\n",
                    observed_max_position,
                    declared_max_position);
        }
        if (p.full_page_tiles) {
            std::fprintf(
                    stderr,
                    "tile_retrieval resident_tokens=%" PRId32 " dense_tokens=%" PRId32
                    " image_token_ratio=%.6f prefix_token_ratio=%.6f yarn_factor=%.3f\n",
                    input_prefix.image_length,
                    dense_image_length,
                    static_cast<double>(input_prefix.image_length) / dense_image_length,
                    static_cast<double>(input_prefix.total_length) /
                            (dense_image_length + prompt_tokens.size()),
                    p.yarn_factor);
        }

        result.image_tokens = input_prefix.image_length;
        result.input_tokens = input_prefix.total_length;
        result.dense_prefix_tokens = input_prefix.total_length;
        result.cached_prefix_tokens = input_prefix.total_length;
        result.physical_prefix_tokens = input_prefix.total_length;
        result.vision_patches = selected_patches;
        result.vision_kept_patches = selected_patches;
        cached_prefix_layout layout = {
            input_prefix.image_length,
            input_prefix.total_length,
            prompt_position,
            input_prefix.generation_position,
        };
        llama_set_d2f_attention(
                context.get(),
                layout.image_length,
                layout.total_length,
                layout.prompt_position,
                layout.generation_position,
                D2F_BLOCK_LENGTH);

        activate_language_affinity();
        llama_perf_context_reset(context.get());
        const auto generation_started = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(context.get()), true);
        if (p.prefix_cache) {
            if (qk_capture) {
                qk_capture->reset();
            }
            const auto prefill_started = std::chrono::steady_clock::now();
            llama_batch & batch = owned_batch->get();
            for (size_t chunk_index = 0; chunk_index < prefix_chunks.size(); ++chunk_index) {
                const lladao::detail::prefix_chunk & chunk = prefix_chunks[chunk_index];
                const bool packed_image_chunk =
                        use_packed_parallel &&
                        chunk.kind == lladao::detail::prefix_chunk_kind::image;
                const int32_t submitted_tokens = chunk.length;
                const auto chunk_started = std::chrono::steady_clock::now();
                if (qk_capture) {
                    const lladao::detail::d2f_qk_capture_phase phase =
                            chunk.kind == lladao::detail::prefix_chunk_kind::image
                                    ? lladao::detail::d2f_qk_capture_phase::image
                                    : chunk.kind == lladao::detail::prefix_chunk_kind::prompt
                                            ? lladao::detail::d2f_qk_capture_phase::prompt
                                            : lladao::detail::d2f_qk_capture_phase::inactive;
                    if (phase == lladao::detail::d2f_qk_capture_phase::inactive) {
                        throw std::logic_error("visual KV compression needs image then prompt prefill calls");
                    }
                    qk_capture->begin_phase(phase, submitted_tokens);
                }
                const auto batch_build_started = std::chrono::steady_clock::now();
                if (packed_image_chunk) {
                    fill_packed_image_prefill_batch(
                            batch,
                            input_prefix,
                            packed_plan,
                            n_embd);
                } else {
                    fill_prefix_batch_range(
                            batch,
                            input_prefix,
                            chunk.offset,
                            chunk.length,
                            n_embd);
                }
                result.prefill_batch_build_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - batch_build_started).count();
                result.prefill_batch_copy_bytes +=
                        static_cast<uint64_t>(submitted_tokens) *
                        (static_cast<uint64_t>(n_embd) * sizeof(float) + sizeof(llama_pos));
                const auto decode_chunk = [&]() {
                    const uint64_t parallel_activation_before = packed_image_chunk
                            ? llama_context_d2f_parallel_activation_count(context.get())
                            : 0;
                    const int32_t status = llama_decode(context.get(), batch);
                    const uint64_t parallel_activation_after = packed_image_chunk
                            ? llama_context_d2f_parallel_activation_count(context.get())
                            : parallel_activation_before;
                    if (parallel_activation_after > parallel_activation_before) {
                        const uint64_t delta = parallel_activation_after - parallel_activation_before;
                        result.parallel_activation_count_delta =
                                delta > std::numeric_limits<uint64_t>::max() -
                                                result.parallel_activation_count_delta
                                        ? std::numeric_limits<uint64_t>::max()
                                        : result.parallel_activation_count_delta + delta;
                    }
                    return status;
                };
                const auto llama_decode_started = std::chrono::steady_clock::now();
                const int32_t decode_result = packed_image_chunk
                        ? lladao::detail::with_packed_prefill_contract(
                                [&]() {
                                    llama_set_d2f_packed_prefill(
                                            context.get(),
                                            packed_plan.streams,
                                            input_prefix.image_length,
                                            packed_plan.lane_tokens);
                                },
                                [&]() noexcept {
                                    llama_set_d2f_packed_prefill(context.get(), 0, 0, 0);
                                },
                                decode_chunk)
                        : decode_chunk();
                if (decode_result == 0) {
                    // Most accelerators enqueue work asynchronously. Synchronize
                    // here so this field measures backend execution rather than
                    // only submission overhead.
                    llama_synchronize(context.get());
                }
                result.prefill_llama_decode_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - llama_decode_started).count();
                if (decode_result != 0) {
                    if (qk_capture) {
                        qk_capture->reset();
                    }
                    if (decode_result == 2 && cancel_requested.load(std::memory_order_relaxed)) {
                        const auto prefill_finished = std::chrono::steady_clock::now();
                        result.prefill_seconds = std::chrono::duration<double>(
                                prefill_finished - prefill_started).count();
                        result.prefill_phase_seconds = std::chrono::duration<double>(
                                prefill_finished - prefill_phase_started).count();
                        result.prefill_resources = lladao::detail::resource_usage_delta(
                                prefill_resources_started,
                                lladao::detail::snapshot_process_resources());
                        print_phase_resources(
                                "prefill", result.prefill_phase_seconds, result.prefill_resources);
                        result.generation_seconds = std::chrono::duration<double>(
                                prefill_finished - generation_started).count();
                        result.cancelled = true;
                        set_cache_audit(result);
                        return result;
                    }
                    throw std::runtime_error(
                            "prefix prefill chunk " + std::to_string(chunk_index + 1) +
                            " llama_decode failed with code " + std::to_string(decode_result));
                }
                if (packed_image_chunk) {
                    llama_memory_t memory = llama_get_memory(context.get());
                    int32_t image_offset = 0;
                    for (int32_t stream = 1; stream < packed_plan.streams; ++stream) {
                        image_offset += packed_plan.lengths[stream - 1];
                        const llama_pos image_position = input_prefix.positions[image_offset];
                        llama_memory_seq_cp(
                                memory,
                                stream,
                                0,
                                image_position,
                                image_position + 1);
                        if (!llama_memory_seq_rm(
                                    memory,
                                    stream,
                                    image_position,
                                    image_position + 1)) {
                            throw std::runtime_error("failed to merge a packed image lane into sequence 0");
                        }
                    }
                }
                if (chunk.kind == lladao::detail::prefix_chunk_kind::image) {
                    ++result.image_prefill_calls;
                }
                if (qk_capture) {
                    qk_capture->finish_phase();
                }
                set_cache_audit(result);
                const double chunk_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - chunk_started).count();
                std::fprintf(
                        stderr,
                        "D2F prefix_prefill mode=%s pack_size=%" PRId32
                        " chunk=%zu/%zu kind=%s component=%" PRId32
                        " tokens=[%" PRId32 ",%" PRId32 ") submitted=%" PRId32
                        " streams=%" PRId32 " lane_tokens=%" PRId32
                        " padding_tokens=%" PRId32 " seconds=%.6f\n",
                        lladao::detail::prefix_prefill_mode_name(base.prefix_prefill_mode),
                        base.prefix_pack_size,
                        chunk_index + 1,
                        prefix_chunks.size(),
                        lladao::detail::prefix_chunk_kind_name(chunk.kind),
                        chunk.component,
                        chunk.offset,
                        chunk.offset + chunk.length,
                        submitted_tokens,
                        packed_image_chunk ? packed_plan.streams : 1,
                        packed_image_chunk ? packed_plan.lane_tokens : chunk.length,
                        0,
                        chunk_seconds);
            }
            const auto prefill_finished = std::chrono::steady_clock::now();
            result.prefill_seconds =
                    std::chrono::duration<double>(prefill_finished - prefill_started).count();
            result.prefill_phase_seconds = std::chrono::duration<double>(
                    prefill_finished - prefill_phase_started).count();
            result.prefill_resources = lladao::detail::resource_usage_delta(
                    prefill_resources_started,
                    lladao::detail::snapshot_process_resources());
            std::fprintf(
                    stderr,
                    "D2F prefill_breakdown prefix_build_seconds=%.6f"
                    " prefix_build_copy_bytes=%" PRIu64
                    " batch_build_seconds=%.6f batch_copy_bytes=%" PRIu64
                    " llama_decode_seconds=%.6f"
                    " prefill_seconds=%.6f phase_seconds=%.6f\n",
                    result.prefix_build_seconds,
                    result.prefix_build_copy_bytes,
                    result.prefill_batch_build_seconds,
                    result.prefill_batch_copy_bytes,
                    result.prefill_llama_decode_seconds,
                    result.prefill_seconds,
                    result.prefill_phase_seconds);
            print_phase_resources(
                    "prefill", result.prefill_phase_seconds, result.prefill_resources);
            result.batched_prefill =
                    parallel_audit.batched_submission &&
                    use_packed_parallel &&
                    result.image_prefill_calls == parallel_audit.image_decode_calls;
            result.parallel_effective =
                    result.batched_prefill &&
                    result.parallel_activation_count_delta > 0;

            std::fprintf(
                    stderr,
                    "D2F parallel_prefill batched_submission=%s parallel_effective=%s"
                    " semantics=%s"
                    " domains=%" PRId32
                    " image_decode_calls=%" PRId32
                    " activation_delta=%" PRIu64
                    " attention_pairs_dense=%" PRIu64
                    " attention_pairs_packed=%" PRIu64
                    " attention_pairs_executed=%" PRIu64
                    " lane_tokens=%" PRId32 " padding_tokens=%" PRId32
                    " backend=%s\n",
                    result.batched_prefill ? "true" : "false",
                    result.parallel_effective ? "true" : "false",
                    result.prefix_prefill_semantics.c_str(),
                    result.domains,
                    result.image_prefill_calls,
                    result.parallel_activation_count_delta,
                    result.attention_pairs_dense,
                    result.attention_pairs_packed,
                    result.attention_pairs_executed,
                    result.parallel_lane_tokens,
                    result.parallel_padding_tokens,
                    result.backend.c_str());

            if (qk_capture) {
                const auto compression_started = std::chrono::steady_clock::now();
                const lladao::detail::d2f_visual_kv_compression_config config = {
                    base.vision_kv_tile_size,
                    base.vision_kv_topk_tiles,
                    base.vision_kv_keep_ratio,
                    base.vision_kv_query_window,
                    base.vision_kv_score_layers,
                    lladao::detail::d2f_kv_score_layer_mode::last,
                    base.vision_kv_pool_kernel,
                };
                const lladao::detail::d2f_visual_kv_keep_plan keep_plan =
                        build_visual_kv_keep_plan_from_capture(
                                *qk_capture,
                                model.get(),
                                visual_prefix_layout,
                                config);
                std::vector<uint32_t> sources(keep_plan.source_indices.size());
                for (size_t source_index = 0; source_index < keep_plan.source_indices.size(); ++source_index) {
                    const int32_t source = keep_plan.source_indices[source_index];
                    if (source < 0 || source >= input_prefix.total_length) {
                        throw std::runtime_error("visual KV keep plan contains an invalid source");
                    }
                    sources[source_index] = static_cast<uint32_t>(source);
                }
                std::vector<llama_pos> positions(
                        keep_plan.destination_positions.begin(),
                        keep_plan.destination_positions.end());
                if (!llama_kv_cache_compact_heads(
                            context.get(),
                            0,
                            static_cast<uint32_t>(input_prefix.total_length),
                            static_cast<uint32_t>(keep_plan.keep_count),
                            sources.data(),
                            sources.size(),
                            positions.data())) {
                    throw std::runtime_error("per-layer/per-head visual KV compaction failed");
                }
                qk_capture->reset();
                result.kv_cache_compression_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - compression_started).count();
                result.cached_prefix_tokens = keep_plan.stats.cached_prefix_tokens;
                result.physical_prefix_tokens = keep_plan.keep_count;
                result.vision_kept_patches = keep_plan.stats.vision_kept_patches;
                result.vision_tiles = keep_plan.stats.vision_tiles;
                result.vision_selected_tiles = keep_plan.stats.vision_selected_tiles;
                result.vision_kv_compression_ratio = keep_plan.stats.compression_ratio;
                layout.image_length = keep_plan.keep_count - prompt_length;
                layout.total_length = keep_plan.keep_count;
                llama_set_d2f_attention(
                        context.get(),
                        layout.image_length,
                        layout.total_length,
                        layout.prompt_position,
                        layout.generation_position,
                        D2F_BLOCK_LENGTH);
                std::fprintf(
                        stderr,
                        "D2F visual_kv_compression dense_prefix=%" PRId32
                        " cached_prefix=%" PRId32 " ratio=%.6f patches=%" PRId32
                        " kept_patches=%" PRId32 " tiles=%" PRId32
                        " selected_tiles=%" PRId32 " seconds=%.6f\n",
                        keep_plan.stats.dense_prefix_tokens,
                        keep_plan.stats.cached_prefix_tokens,
                        keep_plan.stats.compression_ratio,
                        keep_plan.stats.vision_patches,
                        keep_plan.stats.vision_kept_patches,
                        keep_plan.stats.vision_tiles,
                        keep_plan.stats.vision_selected_tiles,
                        result.kv_cache_compression_seconds);
            }
            std::fprintf(
                    stderr,
                    "D2F prefix_cache=enabled mode=%s semantics=%s pack_size=%" PRId32
                    " chunks=%zu image_spans=%" PRId32 " image_prefill_calls=%" PRId32
                    " cached_tokens=%" PRId32 " prefill_seconds=%.6f\n",
                    lladao::detail::prefix_prefill_mode_name(base.prefix_prefill_mode),
                    lladao::detail::prefix_prefill_semantics_name(base.prefix_prefill_mode),
                    base.prefix_pack_size,
                    prefix_chunks.size(),
                    result.image_span_count,
                    result.image_prefill_calls,
                    result.cached_prefix_tokens,
                    result.prefill_seconds);

            if (!pd_state_out.empty()) {
                const auto state_started = std::chrono::steady_clock::now();
                result.state_bytes = save_pd_state(pd_state_out, layout);
                const auto state_finished = std::chrono::steady_clock::now();
                result.state_io_seconds =
                        std::chrono::duration<double>(state_finished - state_started).count();
                result.generation_seconds =
                        std::chrono::duration<double>(state_finished - generation_started).count();
                result.prefilled_only = true;
                std::fprintf(
                        stderr,
                        "D2F PD prefill state=%s bytes=%" PRIu64
                        " state_save_seconds=%.6f total_seconds=%.6f\n",
                        pd_state_out.c_str(),
                        result.state_bytes,
                        result.state_io_seconds,
                        result.generation_seconds);
                if (base.print_timings) {
                    llama_perf_context_print(context.get());
                }
                return result;
            }
        } else {
            std::fprintf(stderr, "D2F prefix_cache=disabled\n");
            result.prefill_phase_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - prefill_phase_started).count();
            result.prefill_resources = lladao::detail::resource_usage_delta(
                    prefill_resources_started,
                    lladao::detail::snapshot_process_resources());
            std::fprintf(
                    stderr,
                    "D2F prefill_breakdown prefix_build_seconds=%.6f"
                    " prefix_build_copy_bytes=%" PRIu64
                    " batch_build_seconds=0.000000 batch_copy_bytes=0"
                    " llama_decode_seconds=0.000000"
                    " prefill_seconds=0.000000 phase_seconds=%.6f\n",
                    result.prefix_build_seconds,
                    result.prefix_build_copy_bytes,
                    result.prefill_phase_seconds);
            print_phase_resources(
                    "prefill", result.prefill_phase_seconds, result.prefill_resources);
        }
        return decode_prefix_locked(
                layout,
                &input_prefix,
                p.prefix_cache,
                std::move(result),
                generation_started);
    }

    d2f_result decode_from_file_locked(const std::string & state_path) {
        if (state_path.empty()) {
            throw std::invalid_argument("PD input path must not be empty");
        }

        const auto state_started = std::chrono::steady_clock::now();
        uint64_t state_bytes = 0;
        const pd_artifact_metadata metadata = load_pd_state(state_path, state_bytes);
        const auto state_finished = std::chrono::steady_clock::now();
        llama_perf_context_reset(context.get());

        d2f_result result;
        set_cache_audit(result);
        result.image_tokens = metadata.layout.image_length;
        result.input_tokens = metadata.layout.total_length;
        result.state_bytes = state_bytes;
        result.state_io_seconds =
                std::chrono::duration<double>(state_finished - state_started).count();
        result.decoded_from_state = true;
        std::fprintf(
                stderr,
                "D2F PD decode state=%s bytes=%" PRIu64
                " state_load_seconds=%.6f vision_model=unloaded\n",
                state_path.c_str(),
                result.state_bytes,
                result.state_io_seconds);

        return decode_prefix_locked(
                metadata.layout,
                nullptr,
                true,
                std::move(result),
                state_finished);
    }
};

d2f_engine::d2f_engine(const d2f_engine_params & params) : impl_(std::make_unique<impl>(params)) {}

d2f_engine::~d2f_engine() = default;

d2f_engine::d2f_engine(d2f_engine &&) noexcept = default;

d2f_engine & d2f_engine::operator=(d2f_engine &&) noexcept = default;

d2f_result d2f_engine::generate(const d2f_request & request) {
    if (!impl_) {
        throw std::logic_error("LLaDA-o engine is not initialized");
    }
    std::lock_guard<std::mutex> lock(impl_->operation_mutex);
    impl::operation_affinity_guard affinity_guard(*impl_);
    impl_->cancel_requested.store(false, std::memory_order_relaxed);
    impl_->active.store(true, std::memory_order_release);
    struct active_guard {
        std::atomic<bool> & active;
        ~active_guard() {
            active.store(false, std::memory_order_release);
        }
    } guard { impl_->active };
    return impl_->generate_locked(request);
}

d2f_result d2f_engine::prefill_to_file(
        const d2f_request & request,
        const std::string & state_path) {
    if (!impl_) {
        throw std::logic_error("LLaDA-o engine is not initialized");
    }
    std::lock_guard<std::mutex> lock(impl_->operation_mutex);
    impl::operation_affinity_guard affinity_guard(*impl_);
    impl_->cancel_requested.store(false, std::memory_order_relaxed);
    impl_->active.store(true, std::memory_order_release);
    struct active_guard {
        std::atomic<bool> & active;
        ~active_guard() {
            active.store(false, std::memory_order_release);
        }
    } guard { impl_->active };
    return impl_->generate_locked(request, state_path);
}

d2f_result d2f_engine::decode_from_file(const std::string & state_path) {
    if (!impl_) {
        throw std::logic_error("LLaDA-o engine is not initialized");
    }
    std::lock_guard<std::mutex> lock(impl_->operation_mutex);
    impl::operation_affinity_guard affinity_guard(*impl_);
    impl_->cancel_requested.store(false, std::memory_order_relaxed);
    impl_->active.store(true, std::memory_order_release);
    struct active_guard {
        std::atomic<bool> & active;
        ~active_guard() {
            active.store(false, std::memory_order_release);
        }
    } guard { impl_->active };
    return impl_->decode_from_file_locked(state_path);
}

void d2f_engine::set_adapter(const std::string & path, float scale) {
    if (!impl_) {
        throw std::logic_error("LLaDA-o engine is not initialized");
    }
    std::lock_guard<std::mutex> lock(impl_->operation_mutex);
    impl_->set_adapter_locked(path, scale);
}

void d2f_engine::clear_adapter() {
    if (!impl_) {
        throw std::logic_error("LLaDA-o engine is not initialized");
    }
    std::lock_guard<std::mutex> lock(impl_->operation_mutex);
    impl_->clear_adapter_locked();
}

void d2f_engine::cancel() noexcept {
    if (impl_) {
        impl_->cancel_requested.store(true, std::memory_order_relaxed);
    }
}

bool d2f_engine::busy() const noexcept {
    return impl_ && impl_->active.load(std::memory_order_acquire);
}

static d2f_request make_request(
        const params & p,
        const std::string & image,
        const std::string & prompt,
        const std::string & retrieval_query,
        const std::string & image_cache_key) {
    d2f_request request;
    request.image_path = image;
    request.image_cache_key = image_cache_key;
    request.prompt = prompt;
    request.retrieval_query = retrieval_query;
    request.full_page_tile_size = p.full_page_tile_size;
    request.tile_retrieval_topk = p.tile_retrieval_topk;
    request.tile_retrieval_mask_rounds = p.tile_retrieval_mask_rounds;
    request.full_page_tiles = p.full_page_tiles;
    request.full_page_overview = p.full_page_overview;
    request.exact_image = p.exact_image;
    request.preprocess_only = p.preprocess_only;
    return request;
}

static json make_phase_resource_output(
        double wall_seconds,
        const d2f_phase_resource_usage & usage) {
    return {
        { "wall_seconds",            wall_seconds                    },
        { "process_cpu_seconds",     usage.process_cpu_seconds       },
        { "minor_page_faults",       usage.minor_page_faults         },
        { "major_page_faults",       usage.major_page_faults         },
        { "inblock",                 usage.block_input_operations    },
        { "oublock",                 usage.block_output_operations   },
        { "proc_io_available",       usage.proc_io_available         },
        { "read_bytes",              usage.read_bytes                },
        { "write_bytes",             usage.write_bytes               },
    };
}

static json make_batch_output(
        const batch_input & input,
        size_t index,
        const d2f_result & result,
        double latency_seconds,
        const std::string & error) {
    const json error_value = error.empty() ? json(nullptr) : json(error);
    return {
        { "id",                    input.id                     },
        { "index",                 index                        },
        { "raw_output",            result.text                  },
        { "language_device_mode",  result.language_device_mode  },
        { "cache_type_k",          result.cache_type_k          },
        { "cache_type_v",          result.cache_type_v          },
        { "flash_attention_mode",  result.flash_attention_mode  },
        { "flash_attention_requested", result.flash_attention_requested },
        { "flash_attention_resolved",  result.flash_attention_resolved  },
        { "prefix_prefill_mode",    result.prefix_prefill_mode    },
        { "prefix_prefill_semantics", result.prefix_prefill_semantics },
        { "backend",                 result.backend                 },
        { "parallel_effective",      result.parallel_effective      },
        { "batched_prefill",         result.batched_prefill         },
        { "attention_pairs_executed_known", result.attention_pairs_executed_known },
        { "parallel_activation_count_delta", result.parallel_activation_count_delta },
        { "domains",                 result.domains                 },
        { "attention_pairs_dense",   result.attention_pairs_dense   },
        { "attention_pairs_packed",  result.attention_pairs_packed  },
        { "attention_pairs_executed", result.attention_pairs_executed },
        { "parallel_lane_tokens",    result.parallel_lane_tokens    },
        { "parallel_padding_tokens", result.parallel_padding_tokens },
        { "prefix_pack_size",       result.prefix_pack_size       },
        { "prefix_chunk_count",     result.prefix_chunk_count     },
        { "image_span_count",       result.image_span_count       },
        { "image_prefill_calls",    result.image_prefill_calls    },
        { "dense_prefix_tokens",    result.dense_prefix_tokens    },
        { "cached_prefix_tokens",   result.cached_prefix_tokens   },
        { "physical_prefix_tokens", result.physical_prefix_tokens },
        { "vision_patches",         result.vision_patches         },
        { "vision_kept_patches",    result.vision_kept_patches    },
        { "vision_tiles",           result.vision_tiles           },
        { "vision_selected_tiles",  result.vision_selected_tiles  },
        { "vision_kv_compression",  result.vision_kv_compression  },
        { "vision_kv_compression_ratio", result.vision_kv_compression_ratio },
        { "cpu_mask_requested",     result.cpu_mask_requested     },
        { "cpu_mask_resolved",      result.cpu_mask_resolved      },
        { "cpu_threads_requested",  result.cpu_threads_requested  },
        { "cpu_threads_resolved",   result.cpu_threads_resolved   },
        { "cpu_strict_requested",   result.cpu_strict_requested   },
        { "cpu_strict_resolved",    result.cpu_strict_resolved    },
        { "cpu_poll_requested",     result.cpu_poll_requested     },
        { "cpu_poll_resolved",      result.cpu_poll_resolved      },
        { "cpu_threadpool_attached", result.cpu_threadpool_attached },
        { "vision_threadpool_shared", result.vision_threadpool_shared },
        { "latency_seconds",       latency_seconds              },
        { "error",                 error_value                  },
        { "iterations",            result.iterations            },
        { "image_tokens",          result.image_tokens          },
        { "input_tokens",          result.input_tokens          },
        { "state_bytes",           result.state_bytes           },
        { "d2f_input_rows",        result.d2f_input_rows        },
        { "d2f_active_rows",       result.d2f_active_rows       },
        { "d2f_rebuild_rows",      result.d2f_rebuild_rows      },
        { "d2f_logit_rows",        result.d2f_logit_rows        },
        { "d2f_reused_input_rows", result.d2f_reused_input_rows },
        { "prefill_seconds",       result.prefill_seconds       },
        { "prefill_phase_seconds", result.prefill_phase_seconds },
        { "prefix_build_seconds",  result.prefix_build_seconds  },
        { "prefix_build_copy_bytes", result.prefix_build_copy_bytes },
        { "prefill_batch_build_seconds", result.prefill_batch_build_seconds },
        { "prefill_batch_copy_bytes", result.prefill_batch_copy_bytes },
        { "prefill_llama_decode_seconds", result.prefill_llama_decode_seconds },
        { "state_io_seconds",      result.state_io_seconds      },
        { "decode_seconds",        result.decode_seconds        },
        { "vision_encode_seconds", result.vision_encode_seconds },
        { "vision_phase_seconds", result.vision_phase_seconds },
        { "vision_embedding_copy_seconds", result.vision_embedding_copy_seconds },
        { "vision_embedding_copy_bytes", result.vision_embedding_copy_bytes },
        { "vision_resources", make_phase_resource_output(
                result.vision_phase_seconds, result.vision_resources) },
        { "prefill_resources", make_phase_resource_output(
                result.prefill_phase_seconds, result.prefill_resources) },
        { "decode_resources", make_phase_resource_output(
                result.decode_seconds, result.decode_resources) },
        { "kv_cache_compression_seconds", result.kv_cache_compression_seconds },
        { "generation_seconds",    result.generation_seconds    },
        { "vision_cache_hit",      result.vision_cache_hit      },
        { "generation_block_cache", result.generation_block_cache },
        { "sparse_generation_logits", result.sparse_generation_logits },
        { "release_vision_after_encode", result.release_vision_after_encode },
        { "vision_context_released", result.vision_context_released },
        { "cancelled",             result.cancelled             },
        { "preprocessed_only",     result.preprocessed_only     },
        { "prefilled_only",        result.prefilled_only        },
        { "decoded_from_state",    result.decoded_from_state    },
    };
}

static int run_batch(
        const params & p,
        const d2f_engine_params & engine_params) {
    const std::vector<batch_input> inputs = read_batch_inputs(p.input_jsonl);
    std::ofstream output(p.output_jsonl, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open batch output: " + p.output_jsonl);
    }

    d2f_engine engine(engine_params);
    bool had_error = false;
    for (size_t index = 0; index < inputs.size(); ++index) {
        const batch_input & input = inputs[index];
        const std::string progress_id = json(input.id).dump(
                -1, ' ', false, json::error_handler_t::replace);
        std::fprintf(
                stderr,
                "batch request %zu/%zu id=%s started\n",
                index + 1,
                inputs.size(),
                progress_id.c_str());
        const auto started = std::chrono::steady_clock::now();
        d2f_result result;
        result.language_device_mode =
                lladao::detail::language_device_mode(engine_params.gpu_layers);
        result.cache_type_k = ggml_type_name(engine_params.cache_type_k);
        result.cache_type_v = ggml_type_name(engine_params.cache_type_v);
        result.flash_attention_requested =
                flash_attention_mode_name(engine_params.flash_attention_mode);
        result.flash_attention_resolved = "not_initialized";
        result.flash_attention_mode = result.flash_attention_resolved;
        result.prefix_prefill_mode =
                lladao::detail::prefix_prefill_mode_name(engine_params.prefix_prefill_mode);
        result.prefix_prefill_semantics =
                lladao::detail::prefix_prefill_semantics_name(engine_params.prefix_prefill_mode);
        result.backend = language_backend_name(engine_params.gpu_layers);
        result.prefix_pack_size = engine_params.prefix_pack_size;
        result.vision_kv_compression = engine_params.vision_kv_compression;
        result.generation_block_cache =
                engine_params.prefix_cache && engine_params.generation_block_cache;
        result.sparse_generation_logits = engine_params.sparse_generation_logits;
        result.cpu_mask_requested =
                lladao::detail::format_cpu_mask(engine_params.cpu_mask);
        result.cpu_mask_resolved = "0x0";
        result.cpu_threads_requested = engine_params.threads;
        result.cpu_threads_resolved = lladao::detail::resolve_cpu_threads(
                engine_params.cpu_mask,
                engine_params.threads);
        result.cpu_strict_requested = engine_params.cpu_strict;
        result.cpu_poll_requested = engine_params.cpu_poll;
        result.vision_threadpool_shared = false;
        result.release_vision_after_encode = engine_params.release_vision_after_encode;
        std::string error;
        try {
            const std::string & retrieval_query =
                    input.retrieval_query.empty() ? p.retrieval_query : input.retrieval_query;
            result = engine.generate(make_request(
                    p,
                    input.image,
                    input.prompt,
                    retrieval_query,
                    input.image_cache_key));
        } catch (const std::exception & exception) {
            error = exception.what();
            had_error = true;
        } catch (...) {
            error = "unknown exception";
            had_error = true;
        }
        const double latency_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
        const json row = make_batch_output(input, index, result, latency_seconds, error);
        output << row.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
        output.flush();
        if (!output) {
            throw std::runtime_error("failed while writing batch output: " + p.output_jsonl);
        }
        std::fprintf(
                stderr,
                "batch request %zu/%zu id=%s status=%s latency_seconds=%.6f\n",
                index + 1,
                inputs.size(),
                progress_id.c_str(),
                error.empty() ? (result.cancelled ? "cancelled" : "ok") : "error",
                latency_seconds);
    }
    return had_error ? 1 : 0;
}

int cli_main(int argc, char ** argv) {
    try {
        const params p = parse_args(argc, argv);
        d2f_engine_params engine_params;
        engine_params.model_path = p.model;
        engine_params.mmproj_path = p.mmproj;
        engine_params.adapter_path = p.lora;
        engine_params.context_size = p.n_ctx;
        engine_params.gpu_layers = p.n_gpu_layers;
        engine_params.devices = p.devices;
        engine_params.threads = p.n_threads;
        engine_params.cpu_mask = p.cpu_mask;
        engine_params.cpu_strict = p.cpu_strict;
        engine_params.cpu_poll = p.cpu_poll;
        engine_params.max_iterations = p.max_iterations;
        engine_params.mask_token_id = p.mask_token_id;
        engine_params.exact_image_max_edge = p.exact_image_max_edge;
        engine_params.prefix_prefill_mode = p.prefix_prefill_mode;
        engine_params.prefix_pack_size = p.prefix_pack_size;
        engine_params.vision_kv_tile_size = p.vision_kv_tile_size;
        engine_params.vision_kv_topk_tiles = p.vision_kv_topk_tiles;
        engine_params.vision_kv_query_window = p.vision_kv_query_window;
        engine_params.vision_kv_score_layers = p.vision_kv_score_layers;
        engine_params.vision_kv_pool_kernel = p.vision_kv_pool_kernel;
        engine_params.vision_kv_keep_ratio = p.vision_kv_keep_ratio;
        engine_params.cache_type_k = p.cache_type_k;
        engine_params.cache_type_v = p.cache_type_v;
        engine_params.flash_attention_mode = p.flash_attention_mode;
        engine_params.yarn_original_context = p.yarn_orig_ctx;
        engine_params.yarn_factor = p.yarn_factor;
        engine_params.vision_gpu = p.vision_gpu;
        engine_params.prefix_cache = p.prefix_cache;
        engine_params.generation_block_cache = p.generation_block_cache;
        engine_params.sparse_generation_logits = p.sparse_generation_logits;
        engine_params.release_vision_after_encode = p.release_vision_after_encode;
        engine_params.vision_kv_compression = p.vision_kv_compression;
        engine_params.use_mmap = p.use_mmap;
        engine_params.print_timings = p.print_timings;

        if (!p.input_jsonl.empty()) {
            return run_batch(p, engine_params);
        }

        const d2f_request request = make_request(p, p.image, p.prompt, p.retrieval_query, {});

        d2f_engine engine(engine_params);
        d2f_result result;
        if (!p.pd_decode_in.empty()) {
            result = engine.decode_from_file(p.pd_decode_in);
        } else if (!p.pd_prefill_out.empty()) {
            result = engine.prefill_to_file(request, p.pd_prefill_out);
        } else {
            result = engine.generate(request);
        }
        if (result.cancelled) {
            return 130;
        }
        if (!result.preprocessed_only && !result.prefilled_only) {
            std::fwrite(result.text.data(), 1, result.text.size(), stdout);
            std::fputc('\n', stdout);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        print_usage(argv[0]);
        return 1;
    }
}

} // namespace lladao
