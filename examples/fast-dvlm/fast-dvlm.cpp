#include "arg.h"
#include "chat.h"
#include "common.h"
#include "fast-dvlm-speculative.h"
#include "ggml.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double elapsed_seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

void show_additional_info(int, char ** argv) {
    std::fprintf(stderr,
                 "Fast-dVLM block-diffusion multimodal CLI\n\n"
                 "Usage: %s -m MODEL --mmproj MMPROJ --image IMAGE -p PROMPT [options]\n\n"
                 "The prompt is causally prefilled with Qwen2.5-VL M-RoPE. Generation uses\n"
                 "Fast-dVLM speculative 32-token blocks (bidirectional draft + causal verify).\n",
                 argv[0]);
}

int32_t read_meta_i32(const llama_model * model, const char * key, int32_t fallback) {
    char value[64] = {};
    if (llama_model_meta_val_str(model, key, value, sizeof(value)) < 0) {
        return fallback;
    }
    char *     end    = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("invalid GGUF metadata value for ") + key + ": " + value);
    }
    return static_cast<int32_t>(parsed);
}

bool read_meta_bool(const llama_model * model, const char * key) {
    char value[16] = {};
    if (llama_model_meta_val_str(model, key, value, sizeof(value)) < 0) {
        return false;
    }
    return std::string(value) == "true" || std::string(value) == "1";
}

llama_token argmax_token(const float * logits, int32_t n_vocab) {
    if (logits == nullptr || n_vocab <= 0) {
        throw std::runtime_error("Fast-dVLM requested a missing logits row");
    }
    return static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
}

struct prompt_timings {
    double    prepare_seconds          = 0.0;
    double    vision_encode_seconds    = 0.0;
    double    language_prefill_seconds = 0.0;
    int32_t   tokens                   = 0;
    llama_pos positions                = 0;
};

prompt_timings evaluate_prompt(mtmd_context *      mctx,
                               llama_context *     lctx,
                               const std::string & formatted_prompt,
                               mtmd::bitmaps &     bitmaps,
                               int32_t             n_batch) {
    prompt_timings timings;
    const auto     prepare_start = clock_type::now();

    mtmd_input_text text = {
        /*.text          =*/formatted_prompt.data(),
        /*.text_len      =*/formatted_prompt.size(),
        /*.add_special   =*/true,
        /*.parse_special =*/true,
    };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto               bitmap_ptrs = bitmaps.c_ptr();
    const int32_t      tokenize_result =
        mtmd_tokenize(mctx, chunks.ptr.get(), &text, bitmap_ptrs.data(), bitmap_ptrs.size());
    if (tokenize_result != 0) {
        throw std::runtime_error("Fast-dVLM multimodal tokenization failed: " + std::to_string(tokenize_result));
    }
    timings.prepare_seconds = elapsed_seconds(prepare_start);
    timings.tokens          = static_cast<int32_t>(mtmd_helper_get_n_tokens(chunks.ptr.get()));

    llama_pos    n_past   = 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk *    chunk      = mtmd_input_chunks_get(chunks.ptr.get(), i);
        const mtmd_input_chunk_type type       = mtmd_input_chunk_get_type(chunk);
        llama_pos                   new_n_past = n_past;

        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            const auto    started = clock_type::now();
            const int32_t result =
                mtmd_helper_eval_chunk_single(mctx, lctx, chunk, n_past, 0, n_batch, i + 1 == n_chunks, &new_n_past);
            timings.language_prefill_seconds += elapsed_seconds(started);
            if (result != 0) {
                throw std::runtime_error("Fast-dVLM text prefill failed: " + std::to_string(result));
            }
        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            auto          started       = clock_type::now();
            const int32_t encode_result = mtmd_encode_chunk(mctx, chunk);
            timings.vision_encode_seconds += elapsed_seconds(started);
            if (encode_result != 0) {
                throw std::runtime_error("Fast-dVLM vision encode failed: " + std::to_string(encode_result));
            }

            float * embeddings = mtmd_get_output_embd(mctx);
            if (embeddings == nullptr) {
                throw std::runtime_error("Fast-dVLM vision encoder returned no embeddings");
            }
            started                     = clock_type::now();
            const int32_t decode_result = mtmd_helper_decode_image_chunk(mctx, lctx, chunk, embeddings, n_past, 0,
                                                                         n_batch, &new_n_past, nullptr, nullptr);
            timings.language_prefill_seconds += elapsed_seconds(started);
            if (decode_result != 0) {
                throw std::runtime_error("Fast-dVLM image-prefix decode failed: " + std::to_string(decode_result));
            }
        } else {
            throw std::runtime_error("Fast-dVLM CLI currently supports one still image and text only");
        }
        n_past = new_n_past;
    }

    timings.positions = n_past;
    return timings;
}

struct generation_result {
    std::vector<llama_token> tokens;
    double                   draft_seconds  = 0.0;
    double                   verify_seconds = 0.0;
    int32_t                  iterations     = 0;
    int32_t                  accepted       = 0;
};

generation_result generate_speculative(llama_context *     lctx,
                                       const llama_vocab * vocab,
                                       llama_pos           prompt_end,
                                       llama_token         first_token,
                                       llama_token         mask_token,
                                       int32_t             block_size,
                                       int32_t             token_shift,
                                       int32_t             max_new_tokens) {
    if (block_size <= 1 || max_new_tokens <= 0) {
        throw std::invalid_argument("Fast-dVLM block and output lengths must be positive");
    }

    generation_result result;
    const int32_t     n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t    memory  = llama_get_memory(lctx);
    llama_batch       batch   = llama_batch_init(block_size, 0, 1);

    auto free_batch = [](llama_batch * value) {
        llama_batch_free(*value);
    };
    std::unique_ptr<llama_batch, decltype(free_batch)> batch_guard(&batch, free_batch);

    if (!llama_vocab_is_eog(vocab, first_token)) {
        result.tokens.push_back(first_token);
    }
    if (llama_vocab_is_eog(vocab, first_token) || max_new_tokens == 1) {
        return result;
    }

    llama_pos   committed_end = prompt_end;
    llama_token inherited     = LLAMA_TOKEN_NULL;
    bool        first_block   = true;

    while (static_cast<int32_t>(result.tokens.size()) < max_new_tokens) {
        std::vector<llama_token> drafted(block_size, mask_token);
        // The first AR token came from prompt prefill and is already emitted.
        // Later inherited tokens are new output and therefore use block_start 0.
        const int32_t            block_start = first_block ? 1 : 0;
        drafted[0]                           = first_block ? first_token : inherited;

        auto fill_batch = [&](const std::vector<llama_token> & tokens) {
            batch.n_tokens = block_size;
            for (int32_t i = 0; i < block_size; ++i) {
                batch.token[i]     = tokens[i];
                batch.pos[i]       = committed_end + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        };

        llama_set_causal_attn(lctx, false);
        fill_batch(drafted);
        auto started = clock_type::now();
        if (llama_decode(lctx, batch) != 0) {
            throw std::runtime_error("Fast-dVLM bidirectional draft decode failed");
        }
        llama_synchronize(lctx);
        result.draft_seconds += elapsed_seconds(started);

        fast_dvlm::apply_draft_predictions(drafted, mask_token, token_shift, [&](int32_t row) {
            return argmax_token(llama_get_logits_ith(lctx, row), n_vocab);
        });

        if (!llama_memory_seq_rm(memory, 0, committed_end, committed_end + block_size)) {
            throw std::runtime_error("Fast-dVLM could not discard draft KV entries");
        }

        llama_set_causal_attn(lctx, true);
        fill_batch(drafted);
        started = clock_type::now();
        if (llama_decode(lctx, batch) != 0) {
            throw std::runtime_error("Fast-dVLM causal verification decode failed");
        }
        llama_synchronize(lctx);
        result.verify_seconds += elapsed_seconds(started);

        std::vector<llama_token> causal_next(block_size);
        for (int32_t i = 0; i < block_size; ++i) {
            causal_next[i] = argmax_token(llama_get_logits_ith(lctx, i), n_vocab);
        }
        const fast_dvlm::speculative_step step = fast_dvlm::resolve_speculative_step(drafted, causal_next, block_start);

        if (step.accepted < block_size &&
            !llama_memory_seq_rm(memory, 0, committed_end + step.accepted, committed_end + block_size)) {
            throw std::runtime_error("Fast-dVLM could not discard rejected verification KV entries");
        }
        committed_end += step.accepted;
        inherited   = step.inherited;
        first_block = false;
        ++result.iterations;
        result.accepted += step.output_count;

        bool stop = false;
        for (llama_token token : step.output) {
            if (llama_vocab_is_eog(vocab, token)) {
                stop = true;
                break;
            }
            result.tokens.push_back(token);
            if (static_cast<int32_t>(result.tokens.size()) >= max_new_tokens) {
                stop = true;
                break;
            }
        }
        if (stop) {
            break;
        }
    }

    return result;
}

}  // namespace

int main(int argc, char ** argv) try {
    std::setlocale(LC_NUMERIC, "C");
    ggml_time_init();
    common_init();

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }
    if (params.mmproj.path.empty() || params.image.size() != 1 || params.prompt.empty()) {
        show_additional_info(argc, argv);
        throw std::invalid_argument("-m, --mmproj, exactly one --image, and -p are required");
    }

    ggml_backend_load_all();
    const auto             total_started = clock_type::now();
    const auto             model_started = clock_type::now();
    common_init_result_ptr llama_init    = common_init_from_params(params);
    llama_model *          model         = llama_init->model();
    llama_context *        lctx          = llama_init->context();
    if (model == nullptr || lctx == nullptr) {
        throw std::runtime_error("failed to initialize Fast-dVLM language model");
    }
    if (!read_meta_bool(model, "fast_dvlm.diffusion")) {
        throw std::runtime_error("GGUF is not marked as a Fast-dVLM diffusion model");
    }
    const llama_vocab * vocab      = llama_model_get_vocab(model);
    const llama_token   mask_token = llama_vocab_mask(vocab);
    if (mask_token == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("Fast-dVLM GGUF has no tokenizer mask token");
    }
    const int32_t block_size  = read_meta_i32(model, "fast_dvlm.block_size", 32);
    const int32_t token_shift = read_meta_i32(model, "fast_dvlm.token_shift", 1);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu             = params.mmproj_use_gpu;
    mparams.print_timings       = true;
    mparams.n_threads           = params.cpuparams.n_threads;
    mparams.flash_attn_type     = params.flash_attn_type;
    mparams.warmup              = params.warmup;
    mparams.image_min_tokens    = params.image_min_tokens;
    mparams.image_max_tokens    = params.image_max_tokens;
    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!mctx) {
        throw std::runtime_error("failed to initialize Fast-dVLM vision model");
    }
    const double model_load_seconds = elapsed_seconds(model_started);

    mtmd::bitmaps                       bitmaps;
    std::vector<mtmd_helper::video_ptr> videos;
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_file(mctx.get(), params.image[0].c_str(), false);
    if (wrapper.bitmap == nullptr) {
        throw std::runtime_error("failed to load Fast-dVLM input image");
    }
    bitmaps.entries.emplace_back(wrapper.bitmap);
    if (wrapper.video_ctx != nullptr) {
        videos.emplace_back(wrapper.video_ctx);
    }

    common_chat_templates_ptr templates = common_chat_templates_init(model, params.chat_template);
    common_chat_msg           message;
    message.role    = "user";
    message.content = mtmd_default_marker() + params.prompt;
    const std::vector<common_chat_msg> history;
    const std::string                  formatted_prompt =
        common_chat_format_single(templates.get(), history, message, true, params.use_jinja);

    llama_set_causal_attn(lctx, true);
    const auto           prefill_started = clock_type::now();
    const prompt_timings prefill         = evaluate_prompt(mctx.get(), lctx, formatted_prompt, bitmaps, params.n_batch);
    const double         prefill_seconds = elapsed_seconds(prefill_started);

    const float *           prompt_logits  = llama_get_logits_ith(lctx, -1);
    const llama_token       first_token    = argmax_token(prompt_logits, llama_vocab_n_tokens(vocab));
    const int32_t           max_new_tokens = params.n_predict < 0 ? 64 : params.n_predict;
    const generation_result generation = generate_speculative(lctx, vocab, prefill.positions, first_token, mask_token,
                                                              block_size, token_shift, max_new_tokens);

    const std::string output        = common_detokenize(lctx, generation.tokens);
    const double      total_seconds = elapsed_seconds(total_started);
    std::printf("FAST_DVLM_OUTPUT_BEGIN\n%s\nFAST_DVLM_OUTPUT_END\n", output.c_str());
    std::fprintf(stderr,
                 "FAST_DVLM_TIMINGS {\"model_load_seconds\":%.6f,\"prompt_prepare_seconds\":%.6f,"
                 "\"vision_encode_seconds\":%.6f,\"language_prefill_seconds\":%.6f,"
                 "\"prefill_seconds\":%.6f,\"draft_seconds\":%.6f,\"verify_seconds\":%.6f,"
                 "\"generation_seconds\":%.6f,\"total_seconds\":%.6f,\"prompt_tokens\":%d,"
                 "\"prompt_positions\":%d,\"output_tokens\":%zu,\"iterations\":%d,"
                 "\"accepted_tokens\":%d,\"block_size\":%d,\"mask_token_id\":%d}\n",
                 model_load_seconds, prefill.prepare_seconds, prefill.vision_encode_seconds,
                 prefill.language_prefill_seconds, prefill_seconds, generation.draft_seconds, generation.verify_seconds,
                 generation.draft_seconds + generation.verify_seconds, total_seconds, prefill.tokens, prefill.positions,
                 generation.tokens.size(), generation.iterations, generation.accepted, block_size, mask_token);
    llama_perf_context_print(lctx);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "llama-fast-dvlm error: %s\n", error.what());
    return 1;
}
