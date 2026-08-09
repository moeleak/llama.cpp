#include "d2f-scheduler.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "lladao-d2f-engine.h"
#include "llama-cpp.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

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

namespace {

constexpr int32_t D2F_GENERATION_LENGTH = 64;
constexpr int32_t D2F_BLOCK_LENGTH      = 16;

struct params {
    std::string model;
    std::string mmproj;
    std::string lora;
    std::string image;
    std::string prompt;
    std::string retrieval_query;
    std::string pd_prefill_out;
    std::string pd_decode_in;
    int32_t     n_ctx          = 16384;
    int32_t     n_gpu_layers   = 999;
    int32_t     n_threads      = 0;
    int32_t     max_iterations = 256;
    int32_t     mask_token_id  = LLAMA_TOKEN_NULL;
    int32_t     full_page_tile_size       = 980;
    int32_t     tile_retrieval_topk       = 0;
    int32_t     tile_retrieval_mask_rounds = 2;
    int32_t     yarn_orig_ctx             = 16384;
    float       yarn_factor               = 1.0f;
    int8_t      vision_gpu     = -1;
    bool        full_page_tiles = false;
    bool        full_page_overview = true;
    bool        preprocess_only = false;
    bool        prefix_cache = true;
    bool        print_timings = true;
};

void print_usage(const char * program) {
    std::fprintf(stderr,
            "Usage: %s --model MODEL.gguf --mmproj MMPROJ.gguf --image IMAGE "
            "--prompt GUI_INSTRUCTION [options]\n"
            "       %s --model MODEL.gguf --pd-decode-in PREFIX.state [options]\n"
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
            "  --vision-gpu          Offload the vision projector even with --gpu-layers 0\n"
            "  --no-vision-gpu       Keep the vision projector on the CPU\n"
            "  --threads N           CPU threads (default: backend choice)\n"
            "  --max-iterations N    D2F iteration limit (default: 256)\n"
            "  --no-prefix-cache     Recompute the complete sequence every D2F iteration\n"
            "  --pd-prefill-out PATH Prefill the exact prefix KV state, save it, and exit\n"
            "  --pd-decode-in PATH   Restore an exact prefix KV state and decode without vision input\n"
            "  --mask-token-id N     Override missing tokenizer mask metadata\n"
            "  --full-page-tiles     Split the original page into exact row-major tiles\n"
            "  --full-page-tile-size N\n"
            "                        Tile edge in pixels (default: 980)\n"
            "  --no-full-page-overview\n"
            "                        Do not append the native-resized whole-page overview\n"
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
        } else if (arg == "--vision-gpu") {
            result.vision_gpu = 1;
        } else if (arg == "--no-vision-gpu") {
            result.vision_gpu = 0;
        } else if (arg == "--threads" || arg == "-t") {
            result.n_threads = parse_i32(arg.c_str(), next());
        } else if (arg == "--max-iterations") {
            result.max_iterations = parse_i32(arg.c_str(), next());
        } else if (arg == "--no-prefix-cache") {
            result.prefix_cache = false;
        } else if (arg == "--pd-prefill-out") {
            result.pd_prefill_out = next();
        } else if (arg == "--pd-decode-in") {
            result.pd_decode_in = next();
        } else if (arg == "--mask-token-id") {
            result.mask_token_id = parse_i32(arg.c_str(), next());
        } else if (arg == "--full-page-tiles") {
            result.full_page_tiles = true;
        } else if (arg == "--full-page-tile-size") {
            result.full_page_tile_size = parse_i32(arg.c_str(), next());
        } else if (arg == "--no-full-page-overview") {
            result.full_page_overview = false;
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
    if (!result.pd_prefill_out.empty() && !result.pd_decode_in.empty()) {
        throw std::invalid_argument("--pd-prefill-out and --pd-decode-in are mutually exclusive");
    }
    if (result.pd_decode_in.empty() &&
        (result.mmproj.empty() || result.image.empty() || result.prompt.empty())) {
        throw std::invalid_argument(
                "--mmproj, --image, and --prompt are required outside PD decode mode");
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
    if (result.full_page_tile_size <= 0 || result.full_page_tile_size > 980) {
        throw std::invalid_argument("--full-page-tile-size must be in [1, 980]");
    }
    if (result.tile_retrieval_topk < 0 || result.tile_retrieval_mask_rounds <= 0) {
        throw std::invalid_argument("retrieval Top-K and mask rounds must be valid");
    }
    if (result.yarn_factor < 1.0f || result.yarn_orig_ctx <= 0) {
        throw std::invalid_argument("YaRN factor and original context must be valid");
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
    int32_t n_tokens = 0;
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
        int32_t n_embd) {
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
    result.embeddings.assign(
            output, output + static_cast<size_t>(result.n_tokens) * static_cast<size_t>(n_embd));
    return result;
}

encoded_image encode_image(
        mtmd_context * mctx,
        const std::string & image_path,
        int32_t n_embd) {
    const bitmap_ptr bitmap = load_image(mctx, image_path);
    return encode_bitmap(mctx, bitmap.get(), n_embd);
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
};

constexpr uint32_t D2F_PD_MAGIC = 0x32445044U;
constexpr int32_t D2F_PD_VERSION = 1;
constexpr size_t D2F_PD_METADATA_TOKENS = 23;

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

    if (metadata.layout.image_length <= 0 ||
        metadata.layout.total_length < metadata.layout.image_length ||
        metadata.layout.prompt_position < 0 ||
        metadata.layout.generation_position < metadata.layout.prompt_position ||
        metadata.layout.total_length > std::numeric_limits<int32_t>::max() - D2F_GENERATION_LENGTH ||
        metadata.n_vocab <= 0 || metadata.n_embd <= 0 ||
        metadata.yarn_orig_ctx <= 0 || metadata.yarn_factor < 1.0f ||
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

double score_image_with_masked_query(
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
        int32_t n_vocab) {
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
    return -loss_sum / static_cast<double>(scored_tokens);
}

struct retrieval_result {
    std::vector<double> scores;
    std::vector<int32_t> selected_source_indices;
    double latency_seconds = 0.0;
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
        int32_t n_vocab) {
    if (topk <= 0) {
        throw std::invalid_argument("tile retrieval Top-K must be positive");
    }

    const auto started = std::chrono::steady_clock::now();
    retrieval_result result;
    result.scores.reserve(source_images.size());
    for (const encoded_image & image : source_images) {
        const double score = score_image_with_masked_query(
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
                n_vocab);
        if (!std::isfinite(score)) {
            throw std::runtime_error("masked-query tile score is not finite");
        }
        result.scores.push_back(score);
        std::fprintf(
                stderr,
                "tile_retrieval score source=%" PRId32 " value=%.8f\n",
                image.source_index,
                score);
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
        int32_t active_generation_length,
        token_embedding_reader & token_embeddings,
        int32_t n_embd) {
    const int32_t n_tokens = input_prefix.total_length + active_generation_length;
    initialize_batch_rows(batch, n_tokens);

    std::copy(input_prefix.embeddings.begin(), input_prefix.embeddings.end(), batch.embd);
    std::copy(input_prefix.positions.begin(), input_prefix.positions.end(), batch.pos);

    for (int32_t i = 0; i < active_generation_length; ++i) {
        const std::vector<float> & embedding = token_embeddings.get(generation_tokens[i]);
        std::copy(
                embedding.begin(),
                embedding.end(),
                batch.embd + static_cast<size_t>(input_prefix.total_length + i) * n_embd);
        batch.pos[input_prefix.total_length + i] = input_prefix.generation_position + i;
    }

    for (int32_t i = 0; i < active_generation_length; ++i) {
        // Same-position D2F decoding only consumes generation logits. Avoid
        // materializing a vocabulary row for every selected visual token.
        batch.logits[input_prefix.total_length + i] = 1;
    }
}

void fill_prefix_batch(
        llama_batch & batch,
        const prefix & input_prefix) {
    initialize_batch_rows(batch, input_prefix.total_length);
    std::copy(input_prefix.embeddings.begin(), input_prefix.embeddings.end(), batch.embd);
    std::copy(input_prefix.positions.begin(), input_prefix.positions.end(), batch.pos);
    batch.logits[input_prefix.total_length - 1] = 1;
}

void fill_generation_batch(
        llama_batch & batch,
        llama_pos generation_position,
        const std::vector<int32_t> & generation_tokens,
        int32_t active_generation_length,
        token_embedding_reader & token_embeddings,
        int32_t n_embd) {
    initialize_batch_rows(batch, active_generation_length);
    for (int32_t i = 0; i < active_generation_length; ++i) {
        const std::vector<float> & embedding = token_embeddings.get(generation_tokens[i]);
        std::copy(
                embedding.begin(),
                embedding.end(),
                batch.embd + static_cast<size_t>(i) * n_embd);
        batch.pos[i] = generation_position + i;
        batch.logits[i] = 1;
    }
}

mtmd::context_ptr make_vision_context(
        const params & p,
        llama_model * model,
        bool exact_tile) {
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = p.vision_gpu < 0 ? p.n_gpu_layers > 0 : p.vision_gpu != 0;
    mtmd_params.print_timings = p.print_timings;
    mtmd_params.lladao_exact_tile = exact_tile;
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
    mtmd::context_ptr native_context = make_vision_context(p, model, false);
    bitmap_ptr page = load_image(native_context.get(), p.image);

    full_page_images result;
    result.width  = static_cast<int32_t>(mtmd_bitmap_get_nx(page.get()));
    result.height = static_cast<int32_t>(mtmd_bitmap_get_ny(page.get()));

    if (p.full_page_overview) {
        result.overview =
                std::make_unique<encoded_image>(
                        encode_bitmap(native_context.get(), page.get(), n_embd));
        result.overview->overview = true;
    }
    native_context.reset();

    const std::vector<tile_box> boxes =
            full_page_tile_boxes(result.width, result.height, p.full_page_tile_size);
    mtmd::context_ptr exact_context = make_vision_context(p, model, true);
    result.sources.reserve(boxes.size());
    llama_pos dense_position = 0;
    for (size_t index = 0; index < boxes.size(); ++index) {
        const bitmap_ptr tile = crop_bitmap(page.get(), boxes[index]);
        encoded_image image = encode_bitmap(exact_context.get(), tile.get(), n_embd);
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

struct d2f_engine::impl {
    params base;
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
    mtmd::context_ptr native_vision;
    llama_adapter_lora_ptr adapter;
    std::string adapter_path;
    float adapter_scale = 1.0f;
    llama_context_ptr context;
    std::unique_ptr<batch_owner> owned_batch;
    int32_t batch_capacity = 0;
    int32_t output_capacity = 0;
    std::atomic<bool> cancel_requested { false };
    std::atomic<bool> active { false };
    std::mutex operation_mutex;

    explicit impl(const d2f_engine_params & p) {
        if (p.model_path.empty()) {
            throw std::invalid_argument("model path must not be empty");
        }
        if (p.context_size <= 0 || p.threads < 0 || p.max_iterations <= 0) {
            throw std::invalid_argument("context size, threads, and max iterations must be valid");
        }
        if (p.yarn_factor < 1.0f || p.yarn_original_context <= 0) {
            throw std::invalid_argument("YaRN factor and original context must be valid");
        }
        if (!std::isfinite(p.adapter_scale)) {
            throw std::invalid_argument("adapter scale must be finite");
        }

        base.model = p.model_path;
        base.mmproj = p.mmproj_path;
        base.n_ctx = p.context_size;
        base.n_gpu_layers = p.gpu_layers;
        base.n_threads = p.threads;
        base.max_iterations = p.max_iterations;
        base.mask_token_id = p.mask_token_id;
        base.yarn_orig_ctx = p.yarn_original_context;
        base.yarn_factor = p.yarn_factor;
        base.vision_gpu = p.vision_gpu;
        base.prefix_cache = p.prefix_cache;
        base.print_timings = p.print_timings;

        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = base.n_gpu_layers;
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

        if (!p.adapter_path.empty()) {
            set_adapter_locked(p.adapter_path, p.adapter_scale);
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

    void ensure_context(int32_t required_batch, int32_t required_outputs) {
        if (context && required_batch <= batch_capacity && required_outputs <= output_capacity) {
            return;
        }

        owned_batch.reset();
        context.reset();

        llama_context_params context_params = llama_context_default_params();
        // The engine-level context size is an upper bound. Allocating a KV cache for
        // that entire limit (16K by default) wastes several GiB on mobile when the
        // current request contains only a few thousand resident tokens. The cache
        // stores one entry per submitted token, so size it to this request instead;
        // sparse multimodal RoPE positions do not require empty cache entries.
        context_params.n_ctx = required_batch;
        context_params.n_batch = required_batch;
        context_params.n_ubatch = required_batch;
        context_params.n_outputs_max = required_outputs;
        context_params.no_perf = false;
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
        std::fprintf(
                stderr,
                "D2F context resident_capacity=%" PRIu32 " request_tokens=%" PRId32
                " configured_limit=%" PRId32 "\n",
                llama_n_ctx(context.get()),
                required_batch,
                base.n_ctx);
        llama_set_causal_attn(context.get(), false);
        apply_adapter();

        owned_batch = std::make_unique<batch_owner>(required_batch, n_embd);
        batch_capacity = required_batch;
        output_capacity = required_outputs;
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
        ensure_context(required_tokens, D2F_GENERATION_LENGTH);
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

            llama_batch & batch = owned_batch->get();
            if (prefix_cache) {
                if (!llama_memory_seq_rm(
                            llama_get_memory(context.get()),
                            0,
                            layout.generation_position,
                            -1)) {
                    throw std::runtime_error("failed to discard the dynamic D2F cache range");
                }
                fill_generation_batch(
                        batch,
                        layout.generation_position,
                        scheduler.tokens(),
                        step.active_end,
                        *token_embeddings,
                        n_embd);
            } else {
                llama_memory_clear(llama_get_memory(context.get()), true);
                fill_batch(
                        batch,
                        *input_prefix,
                        scheduler.tokens(),
                        step.active_end,
                        *token_embeddings,
                        n_embd);
            }
            const int32_t decode_result = llama_decode(context.get(), batch);
            if (decode_result != 0) {
                throw std::runtime_error("llama_decode failed with code " + std::to_string(decode_result));
            }

            const float * logits = llama_get_logits(context.get());
            if (!logits) {
                throw std::runtime_error("language model returned no generation logits");
            }
            const std::vector<diffusion_d2f_candidate> candidates =
                    diffusion_d2f_argmax_candidates(
                            logits,
                            step.active_end,
                            n_vocab,
                            prefix_cache ? 0 : layout.total_length,
                            prefix_cache ? 0 : layout.total_length,
                            D2F_GENERATION_LENGTH,
                            false);
            const std::vector<diffusion_d2f_update> updates = scheduler.apply_candidates(candidates);
            const auto iteration_finished = std::chrono::steady_clock::now();
            ++result.iterations;
            std::fprintf(stderr,
                    "D2F iteration %" PRId32 ": active [%" PRId32 ", %" PRId32
                    "), blocks %" PRId32 ", updates %zu, decode_tokens=%" PRId32
                    ", seconds=%.6f\n",
                    step.iteration,
                    step.active_start,
                    step.active_end,
                    step.blocks_added,
                    updates.size(),
                    batch.n_tokens,
                    std::chrono::duration<double>(iteration_finished - iteration_started).count());
        }

        const auto generation_finished = std::chrono::steady_clock::now();
        result.decode_seconds =
                std::chrono::duration<double>(generation_finished - decode_started).count();
        result.generation_seconds =
                std::chrono::duration<double>(generation_finished - generation_started).count();
        std::fprintf(
                stderr,
                "D2F decode_seconds=%.6f generation_seconds=%.6f\n",
                result.decode_seconds,
                result.generation_seconds);
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

        const std::string operation = trim_text(p.prompt);
        const std::string retrieval_query =
                trim_text(p.retrieval_query.empty() ? operation : p.retrieval_query);
        if (operation.empty() || retrieval_query.empty()) {
            throw std::invalid_argument("GUI operation and retrieval query must not be blank");
        }

        std::vector<encoded_image> native_images;
        full_page_images page;
        std::vector<const encoded_image *> selected_images;
        std::vector<llama_token> prompt_tokens;
        llama_pos prompt_position = 1;
        int32_t dense_image_length = 0;
        int32_t max_work_tokens = 0;
        std::vector<llama_token> retrieval_query_tokens;

        if (p.full_page_tiles) {
            native_vision.reset();
            page = encode_full_page(p, model.get(), n_embd);
            if (page.sources.empty()) {
                throw std::runtime_error("full-page tiling produced no source images");
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
            if (!native_vision) {
                native_vision = make_vision_context(base, model.get(), false);
            }
            native_images.push_back(encode_image(native_vision.get(), p.image, n_embd));
            native_images.back().dense_position = 0;
            selected_images.push_back(&native_images.back());
            dense_image_length = native_images.back().n_tokens + 2;
            max_work_tokens =
                    dense_image_length +
                    static_cast<int32_t>(prompt_tokens.size()) +
                    D2F_GENERATION_LENGTH;
        }

        if (max_work_tokens > p.n_ctx) {
            throw std::runtime_error(
                    "the largest scoring or generation batch needs " +
                    std::to_string(max_work_tokens) +
                    " tokens but context size is " + std::to_string(p.n_ctx));
        }

        d2f_result result;
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
            result.preprocessed_only = true;
            return result;
        }

        const int32_t required_outputs = std::max<int32_t>(
                D2F_GENERATION_LENGTH,
                retrieval_query_tokens.empty()
                        ? 0
                        : static_cast<int32_t>(retrieval_query_tokens.size()));
        ensure_context(max_work_tokens, required_outputs);

        if (p.tile_retrieval_topk > 0) {
            llama_set_d2f_attention(context.get(), -1, -1, -1, -1, 0);
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
                    n_vocab);

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

        const prefix input_prefix = build_prefix(
                *token_embeddings,
                selected_images,
                prompt_tokens,
                vision_start,
                vision_end,
                prompt_position,
                n_embd);
        const int32_t max_tokens = input_prefix.total_length + D2F_GENERATION_LENGTH;
        if (max_tokens > p.n_ctx || max_tokens > max_work_tokens) {
            throw std::logic_error("selected prefix exceeds its validated capacity");
        }
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
        const cached_prefix_layout layout = {
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

        llama_perf_context_reset(context.get());
        const auto generation_started = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(context.get()), true);
        if (p.prefix_cache) {
            const auto prefill_started = std::chrono::steady_clock::now();
            llama_batch & batch = owned_batch->get();
            fill_prefix_batch(batch, input_prefix);
            const int32_t decode_result = llama_decode(context.get(), batch);
            if (decode_result != 0) {
                throw std::runtime_error("prefix prefill llama_decode failed with code " + std::to_string(decode_result));
            }
            llama_synchronize(context.get());
            const auto prefill_finished = std::chrono::steady_clock::now();
            result.prefill_seconds =
                    std::chrono::duration<double>(prefill_finished - prefill_started).count();
            std::fprintf(
                    stderr,
                    "D2F prefix_cache=enabled cached_tokens=%" PRId32 " prefill_seconds=%.6f\n",
                    input_prefix.total_length,
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

int cli_main(int argc, char ** argv) {
    try {
        const params p = parse_args(argc, argv);
        d2f_engine_params engine_params;
        engine_params.model_path = p.model;
        engine_params.mmproj_path = p.mmproj;
        engine_params.adapter_path = p.lora;
        engine_params.context_size = p.n_ctx;
        engine_params.gpu_layers = p.n_gpu_layers;
        engine_params.threads = p.n_threads;
        engine_params.max_iterations = p.max_iterations;
        engine_params.mask_token_id = p.mask_token_id;
        engine_params.yarn_original_context = p.yarn_orig_ctx;
        engine_params.yarn_factor = p.yarn_factor;
        engine_params.vision_gpu = p.vision_gpu;
        engine_params.prefix_cache = p.prefix_cache;

        d2f_request request;
        request.image_path = p.image;
        request.prompt = p.prompt;
        request.retrieval_query = p.retrieval_query;
        request.full_page_tile_size = p.full_page_tile_size;
        request.tile_retrieval_topk = p.tile_retrieval_topk;
        request.tile_retrieval_mask_rounds = p.tile_retrieval_mask_rounds;
        request.full_page_tiles = p.full_page_tiles;
        request.full_page_overview = p.full_page_overview;
        request.preprocess_only = p.preprocess_only;

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
