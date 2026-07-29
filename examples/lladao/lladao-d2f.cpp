#include "d2f-scheduler.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-cpp.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
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
    int32_t     n_ctx          = 16384;
    int32_t     n_gpu_layers   = 999;
    int32_t     n_threads      = 0;
    int32_t     max_iterations = 256;
    int32_t     mask_token_id  = LLAMA_TOKEN_NULL;
    bool        preprocess_only = false;
};

void print_usage(const char * program) {
    std::fprintf(stderr,
            "Usage: %s --model MODEL.gguf --mmproj MMPROJ.gguf --image IMAGE "
            "--prompt GUI_INSTRUCTION [options]\n"
            "\n"
            "Required:\n"
            "  --model PATH          LLaDA-o language GGUF\n"
            "  --mmproj PATH         LLaDA-o vision projector GGUF\n"
            "  --image PATH          Screenshot to ground\n"
            "  --prompt TEXT         Exact GUI instruction, e.g. 'Click on Settings.'\n"
            "\n"
            "Options:\n"
            "  --lora PATH           Optional F32 D2F LoRA GGUF\n"
            "  --ctx-size N          Context capacity (default: 16384)\n"
            "  --gpu-layers N        Layers to offload (default: 999)\n"
            "  --threads N           CPU threads (default: backend choice)\n"
            "  --max-iterations N    D2F iteration limit (default: 256)\n"
            "  --mask-token-id N     Override missing tokenizer mask metadata\n"
            "  --preprocess-only     Stop after validating multimodal layout\n"
            "  -h, --help            Show this help\n",
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
        } else if (arg == "--threads" || arg == "-t") {
            result.n_threads = parse_i32(arg.c_str(), next());
        } else if (arg == "--max-iterations") {
            result.max_iterations = parse_i32(arg.c_str(), next());
        } else if (arg == "--mask-token-id") {
            result.mask_token_id = parse_i32(arg.c_str(), next());
        } else if (arg == "--preprocess-only") {
            result.preprocess_only = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (result.model.empty() || result.mmproj.empty() || result.image.empty() || result.prompt.empty()) {
        throw std::invalid_argument("--model, --mmproj, --image, and --prompt are required");
    }
    if (result.n_ctx <= 0 || result.n_threads < 0 || result.max_iterations <= 0) {
        throw std::invalid_argument("context size, threads, and max iterations must be valid");
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
};

encoded_image encode_image(
        mtmd_context * mctx,
        const std::string & image_path,
        int32_t n_embd) {
    mtmd_helper_bitmap_wrapper wrapper =
            mtmd_helper_bitmap_init_from_file(mctx, image_path.c_str(), false);
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(wrapper.bitmap, mtmd_bitmap_free);
    std::unique_ptr<mtmd_helper_video, decltype(&mtmd_helper_video_free)>
            video(wrapper.video_ctx, mtmd_helper_video_free);
    if (!bitmap) {
        throw std::runtime_error("failed to load screenshot: " + image_path);
    }
    if (mtmd_bitmap_is_audio(bitmap.get())) {
        throw std::runtime_error("LLaDA-o requires an image, not audio");
    }

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (!chunks.ptr) {
        throw std::runtime_error("failed to allocate multimodal chunks");
    }
    const mtmd_bitmap * bitmaps[] = { bitmap.get() };
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

struct prefix {
    std::vector<float> embeddings;
    std::vector<llama_pos> positions;
    int32_t image_length = 0;
    int32_t total_length = 0;
    llama_pos generation_position = 0;
};

void append_embedding(std::vector<float> & destination, const std::vector<float> & source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

prefix build_prefix(
        token_embedding_reader & token_embeddings,
        const encoded_image & image,
        const std::vector<llama_token> & prompt_tokens,
        llama_token vision_start,
        llama_token vision_end,
        int32_t n_embd) {
    prefix result;
    const size_t prefix_tokens =
            static_cast<size_t>(image.n_tokens) + 2 + prompt_tokens.size();
    result.embeddings.reserve(prefix_tokens * static_cast<size_t>(n_embd));
    result.positions.reserve(prefix_tokens);

    append_embedding(result.embeddings, token_embeddings.get(vision_start));
    result.positions.push_back(0);
    result.embeddings.insert(result.embeddings.end(), image.embeddings.begin(), image.embeddings.end());
    result.positions.insert(result.positions.end(), static_cast<size_t>(image.n_tokens), 0);
    append_embedding(result.embeddings, token_embeddings.get(vision_end));
    result.positions.push_back(0);
    result.image_length = image.n_tokens + 2;

    llama_pos position = 1;
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

void fill_batch(
        llama_batch & batch,
        const prefix & input_prefix,
        const std::vector<int32_t> & generation_tokens,
        int32_t active_generation_length,
        token_embedding_reader & token_embeddings,
        int32_t n_embd) {
    const int32_t n_tokens = input_prefix.total_length + active_generation_length;
    batch.n_tokens = n_tokens;

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

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        // Diffusion models expose one logits row for every input position.
        // Keep that layout explicit and index generation rows after the
        // multimodal prefix below.
        batch.logits[i] = 1;
    }
}

void print_layout(
        const prefix & input_prefix,
        int32_t n_image_patches,
        size_t n_prompt_tokens,
        int32_t n_ctx) {
    std::fprintf(stderr,
            "LLaDA-o input layout:\n"
            "  image split: %" PRId32 " tokens "
            "(vision_start + %" PRId32 " patches + vision_end), shared RoPE position 0\n"
            "  text split:  %zu tokens (BOS + GUI instruction + EOS), RoPE positions 1..%" PRId32 "\n"
            "  D2F output:   %" PRId32 " masks, block length %" PRId32 ", RoPE positions %" PRId32 "..%" PRId32 "\n"
            "  active max:   %" PRId32 " tokens, context capacity %" PRId32 "\n",
            input_prefix.image_length,
            n_image_patches,
            n_prompt_tokens,
            input_prefix.generation_position - 1,
            D2F_GENERATION_LENGTH,
            D2F_BLOCK_LENGTH,
            input_prefix.generation_position,
            input_prefix.generation_position + D2F_GENERATION_LENGTH - 1,
            input_prefix.total_length + D2F_GENERATION_LENGTH,
            n_ctx);
}

int run(const params & p) {
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = p.n_gpu_layers;
    llama_model_ptr model(llama_model_load_from_file(p.model.c_str(), model_params));
    if (!model) {
        throw std::runtime_error("failed to load language model: " + p.model);
    }
    if (!llama_model_is_diffusion(model.get())) {
        throw std::runtime_error("language GGUF is not marked as a diffusion model");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_embd = llama_model_n_embd_inp(model.get());
    if (n_vocab <= 0 || n_embd <= 0) {
        throw std::runtime_error("language model has invalid vocabulary or embedding width");
    }

    const llama_token bos = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    llama_token mask = llama_vocab_mask(vocab);
    if (mask == LLAMA_TOKEN_NULL) {
        mask = p.mask_token_id;
    }
    if (bos == LLAMA_TOKEN_NULL || eos == LLAMA_TOKEN_NULL || mask == LLAMA_TOKEN_NULL) {
        throw std::runtime_error(
                "model must provide BOS, EOS, and MASK tokens (or pass --mask-token-id)");
    }
    const llama_token vision_start = special_token(vocab, "<|vision_start|>");
    const llama_token vision_end   = special_token(vocab, "<|vision_end|>");

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = p.n_gpu_layers > 0;
    mtmd_params.print_timings = true;
    if (p.n_threads > 0) {
        mtmd_params.n_threads = p.n_threads;
    }
    mtmd::context_ptr mctx(mtmd_init_from_file(p.mmproj.c_str(), model.get(), mtmd_params));
    if (!mctx) {
        throw std::runtime_error("failed to load LLaDA-o mmproj: " + p.mmproj);
    }
    if (!mtmd_support_vision(mctx.get())) {
        throw std::runtime_error("mmproj does not support vision");
    }

    const encoded_image image = encode_image(mctx.get(), p.image, n_embd);
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.push_back(bos);
    const std::vector<llama_token> instruction = tokenize(vocab, p.prompt, false, false);
    prompt_tokens.insert(prompt_tokens.end(), instruction.begin(), instruction.end());
    prompt_tokens.push_back(eos);

    token_embedding_reader token_embeddings(p.model, n_embd, n_vocab);
    const prefix input_prefix = build_prefix(
            token_embeddings, image, prompt_tokens, vision_start, vision_end, n_embd);
    const int32_t max_tokens = input_prefix.total_length + D2F_GENERATION_LENGTH;
    if (max_tokens > p.n_ctx) {
        throw std::runtime_error(
                "input needs " + std::to_string(max_tokens) +
                " tokens but --ctx-size is " + std::to_string(p.n_ctx));
    }
    print_layout(input_prefix, image.n_tokens, prompt_tokens.size(), p.n_ctx);
    if (p.preprocess_only) {
        return 0;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = p.n_ctx;
    context_params.n_batch = max_tokens;
    context_params.n_ubatch = max_tokens;
    context_params.no_perf = false;
    if (p.n_threads > 0) {
        context_params.n_threads = p.n_threads;
        context_params.n_threads_batch = p.n_threads;
    }
    llama_context_ptr context(llama_init_from_model(model.get(), context_params));
    if (!context) {
        throw std::runtime_error("failed to create language context");
    }

    llama_adapter_lora_ptr lora;
    if (!p.lora.empty()) {
        lora.reset(llama_adapter_lora_init(model.get(), p.lora.c_str()));
        if (!lora) {
            throw std::runtime_error("failed to load LoRA: " + p.lora);
        }
        llama_adapter_lora * adapters[] = { lora.get() };
        float scales[] = { 1.0f };
        if (llama_set_adapters_lora(context.get(), adapters, 1, scales) != 0) {
            throw std::runtime_error("failed to apply LoRA");
        }
    }

    llama_set_causal_attn(context.get(), false);
    llama_set_d2f_attention(
            context.get(), input_prefix.image_length, input_prefix.total_length, D2F_BLOCK_LENGTH);

    diffusion_d2f_scheduler_params scheduler_params;
    scheduler_params.generation_length = D2F_GENERATION_LENGTH;
    scheduler_params.block_length = D2F_BLOCK_LENGTH;
    scheduler_params.max_iterations = p.max_iterations;
    scheduler_params.mask_token_id = mask;
    scheduler_params.initial_token_id = bos;
    scheduler_params.eos_token_id = eos;
    diffusion_d2f_scheduler scheduler(scheduler_params);

    batch_owner owned_batch(max_tokens, n_embd);
    while (!scheduler.done()) {
        const diffusion_d2f_step step = scheduler.prepare_step();
        if (step.done) {
            break;
        }

        llama_memory_clear(llama_get_memory(context.get()), true);
        llama_batch & batch = owned_batch.get();
        fill_batch(
                batch,
                input_prefix,
                scheduler.tokens(),
                step.active_end,
                token_embeddings,
                n_embd);
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
                        input_prefix.total_length + step.active_end,
                        n_vocab,
                        0,
                        input_prefix.total_length,
                        D2F_GENERATION_LENGTH,
                        false);
        const std::vector<diffusion_d2f_update> updates = scheduler.apply_candidates(candidates);
        std::fprintf(stderr,
                "D2F iteration %" PRId32 ": active [%" PRId32 ", %" PRId32
                "), blocks %" PRId32 ", updates %zu\n",
                step.iteration,
                step.active_start,
                step.active_end,
                step.blocks_added,
                updates.size());
    }

    const std::vector<int32_t> & generated = scheduler.tokens();
    const int32_t output_length = scheduler.output_length();
    for (int32_t i = 0; i < output_length; ++i) {
        if (generated[i] == mask) {
            throw std::runtime_error("D2F stopped with unresolved mask tokens");
        }
        const std::string piece = token_piece(vocab, generated[i]);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
    }
    std::fputc('\n', stdout);
    llama_perf_context_print(context.get());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const params p = parse_args(argc, argv);
        return run(p);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        print_usage(argv[0]);
        return 1;
    }
}
