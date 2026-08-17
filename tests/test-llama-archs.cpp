#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "d2f-scheduler.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "lladao-d2f-decode.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-v/--verbose]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 128;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    }

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2)); // sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static double test_d2f_prefix_cache(llama_model * model, const std::vector<llama_token> & tokens) {
    static const std::vector<llama_pos> positions = { 0, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8 };
    static constexpr int32_t image_prefix_length = 6;
    static constexpr int32_t prefix_length = 8;
    static constexpr int32_t prompt_position = 3;
    static constexpr int32_t generation_position = 5;
    static constexpr int32_t block_size = 2;
    static constexpr int32_t generation_length = 4;

    GGML_ASSERT(tokens.size() >= positions.size());

    auto make_context = [model](uint32_t n_ubatch = 16) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 16;
        params.n_batch = n_ubatch;
        params.n_ubatch = n_ubatch;
        params.n_outputs_max = generation_length;
        params.no_perf = true;

        llama_context_ptr context(llama_init_from_model(model, params));
        if (!context) {
            throw std::runtime_error("failed to create D2F cache test context");
        }
        if (llama_context_flash_attn_type(context.get()) == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            throw std::runtime_error("D2F context did not expose resolved Flash Attention mode");
        }
        llama_set_causal_attn(context.get(), false);
        llama_set_d2f_attention(
                context.get(),
                image_prefix_length,
                prefix_length,
                prompt_position,
                generation_position,
                block_size);
        return context;
    };

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> logits_full;
    logits_full.reserve(generation_length*n_vocab);
    {
        llama_context_ptr context = make_context();
        llama_batch batch = llama_batch_init(positions.size(), 0, 1);
        for (int32_t i = 0; i < (int32_t) positions.size(); ++i) {
            common_batch_add(batch, tokens[i], positions[i], { 0 }, i >= prefix_length);
        }
        if (llama_encode(context.get(), batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("D2F full-sequence encode failed");
        }
        for (int32_t i = prefix_length; i < (int32_t) positions.size(); ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            logits_full.insert(logits_full.end(), row, row + n_vocab);
        }
        llama_batch_free(batch);
    }

    std::vector<float> logits_cached;
    logits_cached.reserve(generation_length*n_vocab);
    {
        llama_context_ptr context = make_context();
        llama_batch prefix = llama_batch_init(prefix_length, 0, 1);
        for (int32_t i = 0; i < prefix_length; ++i) {
            common_batch_add(prefix, tokens[i], positions[i], { 0 }, i == prefix_length - 1);
        }
        if (llama_decode(context.get(), prefix) != 0) {
            llama_batch_free(prefix);
            throw std::runtime_error("D2F cached prefix decode failed");
        }
        llama_batch_free(prefix);

        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F cached generation decode failed");
        }
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            logits_cached.insert(logits_cached.end(), row, row + n_vocab);
        }
        llama_batch_free(generation);
    }

    std::vector<float> logits_component_parallel;
    logits_component_parallel.reserve(generation_length*n_vocab);
    {
        llama_context_ptr context = make_context(image_prefix_length);
        llama_batch image = llama_batch_init(image_prefix_length, 0, 1);
        for (int32_t i = 0; i < image_prefix_length; ++i) {
            common_batch_add(image, tokens[i], positions[i], { 0 }, i == image_prefix_length - 1);
        }
        if (llama_decode(context.get(), image) != 0) {
            llama_batch_free(image);
            throw std::runtime_error("D2F parallel component image decode failed");
        }
        llama_batch_free(image);

        const int32_t prompt_length = prefix_length - image_prefix_length;
        llama_batch prompt = llama_batch_init(prompt_length, 0, 1);
        for (int32_t i = 0; i < prompt_length; ++i) {
            const int32_t source = image_prefix_length + i;
            common_batch_add(prompt, tokens[source], positions[source], { 0 }, i == prompt_length - 1);
        }
        if (llama_decode(context.get(), prompt) != 0) {
            llama_batch_free(prompt);
            throw std::runtime_error("D2F parallel component prompt decode failed");
        }
        llama_batch_free(prompt);

        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F parallel component generation decode failed");
        }
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            logits_component_parallel.insert(logits_component_parallel.end(), row, row + n_vocab);
        }
        llama_batch_free(generation);
    }

    const int32_t packed_test_n_embd = llama_model_n_embd_inp(model);
    static const std::vector<llama_pos> packed_test_positions = { 0, 1, 1, 2, 2, 2 };
    static constexpr int32_t packed_test_streams = 3;
    static constexpr int32_t packed_test_offsets[packed_test_streams] = { 0, 1, 3 };
    static constexpr int32_t packed_test_lengths[packed_test_streams] = { 1, 2, 3 };
    std::vector<float> packed_test_image_embeddings(
            static_cast<size_t>(image_prefix_length)*packed_test_n_embd);
    for (int32_t token = 0; token < image_prefix_length; ++token) {
        for (int32_t dim = 0; dim < packed_test_n_embd; ++dim) {
            packed_test_image_embeddings[static_cast<size_t>(token)*packed_test_n_embd + dim] =
                    static_cast<float>((tokens[token] + 3*dim + token) % 29)/29.0f;
        }
    }

    {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 16;
        params.n_batch = image_prefix_length;
        params.n_ubatch = image_prefix_length;
        params.n_outputs_max = generation_length;
        params.n_seq_max = packed_test_streams;
        params.kv_unified = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.no_perf = true;
        llama_context_ptr context(llama_init_from_model(model, params));
        if (!context) {
            throw std::runtime_error("failed to create non-Flash D2F packed contract context");
        }

        // Clearing is always valid, including on a context without Flash Attention.
        llama_set_d2f_packed_prefill(context.get(), 0, 0, 0);
        bool rejected = false;
        try {
            llama_set_d2f_packed_prefill(
                    context.get(),
                    packed_test_streams,
                    image_prefix_length,
                    3);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("packed D2F prefill accepted a context without Flash Attention");
        }
    }

    std::vector<float> logits_packed_reference;
    logits_packed_reference.reserve(generation_length*n_vocab);
    {
        llama_context_ptr context = make_context(image_prefix_length);
        for (int32_t stream = 0; stream < packed_test_streams; ++stream) {
            const int32_t offset = packed_test_offsets[stream];
            const int32_t length = packed_test_lengths[stream];
            llama_batch image = llama_batch_init(length, packed_test_n_embd, 1);
            image.n_tokens = length;
            for (int32_t i = 0; i < length; ++i) {
                const int32_t source = offset + i;
                std::copy_n(
                        packed_test_image_embeddings.data() + static_cast<size_t>(source)*packed_test_n_embd,
                        packed_test_n_embd,
                        image.embd + static_cast<size_t>(i)*packed_test_n_embd);
                image.pos[i] = packed_test_positions[source];
                image.n_seq_id[i] = 1;
                image.seq_id[i][0] = 0;
                image.logits[i] = i == length - 1;
            }
            if (llama_decode(context.get(), image) != 0) {
                llama_batch_free(image);
                throw std::runtime_error("D2F component-exact image decode failed");
            }
            llama_batch_free(image);
        }

        const int32_t prompt_length = prefix_length - image_prefix_length;
        llama_batch prompt = llama_batch_init(prompt_length, 0, 1);
        for (int32_t i = 0; i < prompt_length; ++i) {
            const int32_t source = image_prefix_length + i;
            common_batch_add(prompt, tokens[source], positions[source], { 0 }, i == prompt_length - 1);
        }
        if (llama_decode(context.get(), prompt) != 0) {
            llama_batch_free(prompt);
            throw std::runtime_error("D2F packed-lane reference prompt decode failed");
        }
        llama_batch_free(prompt);

        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F packed-lane reference generation decode failed");
        }
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            logits_packed_reference.insert(logits_packed_reference.end(), row, row + n_vocab);
        }
        llama_batch_free(generation);
    }

    std::vector<float> logits_packed_lanes;
    logits_packed_lanes.reserve(generation_length*n_vocab);
    {
        static constexpr int32_t packed_tokens = image_prefix_length;

        llama_context_params params = llama_context_default_params();
        params.n_ctx = 16;
        params.n_batch = packed_tokens;
        params.n_ubatch = packed_tokens;
        params.n_outputs_max = generation_length;
        params.n_seq_max = 4;
        params.kv_unified = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.no_perf = true;
        llama_context_ptr context(llama_init_from_model(model, params));
        if (!context) {
            throw std::runtime_error("failed to create D2F packed-lane context");
        }
        if (llama_context_d2f_parallel_activation_count(context.get()) != 0) {
            throw std::runtime_error("CPU context reported a CUDA D2F parallel activation");
        }
        if (context->d2f_packed_prefill_stream_capacity_max() != 0) {
            throw std::runtime_error("packed stream capacity was nonzero at context initialization");
        }
        llama_set_causal_attn(context.get(), false);
        llama_set_d2f_attention(
                context.get(),
                image_prefix_length,
                prefix_length,
                prompt_position,
                generation_position,
                block_size);
        llama_set_d2f_packed_prefill(context.get(), packed_test_streams, packed_tokens, 3);
        if (context->d2f_packed_prefill_stream_capacity_max() != packed_test_streams) {
            throw std::runtime_error("packed stream capacity did not record its first high-water mark");
        }
        llama_set_d2f_packed_prefill(context.get(), 0, 0, 0);
        if (context->d2f_packed_prefill_stream_capacity_max() != packed_test_streams) {
            throw std::runtime_error("clearing packed prefill lowered its stream capacity");
        }
        llama_set_d2f_packed_prefill(context.get(), 2, packed_tokens, 3);
        if (context->d2f_packed_prefill_stream_capacity_max() != packed_test_streams) {
            throw std::runtime_error("a smaller packed request changed the stream capacity");
        }
        llama_set_d2f_packed_prefill(context.get(), 0, 0, 0);
        llama_set_d2f_packed_prefill(context.get(), 4, packed_tokens, 3);
        if (context->d2f_packed_prefill_stream_capacity_max() != 4) {
            throw std::runtime_error("a larger packed request did not raise the stream capacity");
        }
        llama_set_d2f_packed_prefill(context.get(), 0, 0, 0);
        llama_set_d2f_packed_prefill(context.get(), packed_test_streams, packed_tokens, 3);

        llama_batch invalid_image = llama_batch_init(packed_tokens, packed_test_n_embd, 1);
        invalid_image.n_tokens = packed_tokens;
        for (int32_t packed_index = 0; packed_index < packed_tokens; ++packed_index) {
            std::copy_n(
                    packed_test_image_embeddings.data() +
                            static_cast<size_t>(packed_index)*packed_test_n_embd,
                    packed_test_n_embd,
                    invalid_image.embd + static_cast<size_t>(packed_index)*packed_test_n_embd);
            invalid_image.pos[packed_index] = packed_test_positions[packed_index];
            invalid_image.n_seq_id[packed_index] = 1;
            invalid_image.seq_id[packed_index][0] = packed_index < packed_test_offsets[1] ? 0 : 1;
            invalid_image.logits[packed_index] = packed_index == packed_tokens - 1;
        }
        if (llama_decode(context.get(), invalid_image) == 0) {
            llama_batch_free(invalid_image);
            throw std::runtime_error("D2F packed-lane layout mismatch fell back to dense attention");
        }
        llama_batch_free(invalid_image);

        llama_batch image = llama_batch_init(packed_tokens, packed_test_n_embd, 1);
        image.n_tokens = packed_tokens;
        for (int32_t packed_index = 0; packed_index < packed_tokens; ++packed_index) {
            std::copy_n(
                    packed_test_image_embeddings.data() +
                            static_cast<size_t>(packed_index)*packed_test_n_embd,
                    packed_test_n_embd,
                    image.embd + static_cast<size_t>(packed_index)*packed_test_n_embd);
            image.pos[packed_index] = packed_test_positions[packed_index];
            image.n_seq_id[packed_index] = 1;
            image.seq_id[packed_index][0] = packed_index < packed_test_offsets[1]
                    ? 0
                    : packed_index < packed_test_offsets[2] ? 1 : 2;
            image.logits[packed_index] = packed_index == packed_tokens - 1;
        }
        if (llama_decode(context.get(), image) != 0) {
            llama_batch_free(image);
            throw std::runtime_error("D2F packed-lane image decode failed");
        }
        if (llama_context_d2f_parallel_activation_count(context.get()) != 0) {
            llama_batch_free(image);
            throw std::runtime_error("CPU packed decode reported hardware parallel activation");
        }
        llama_batch_free(image);
        llama_set_d2f_packed_prefill(context.get(), 0, 0, 0);
        for (int32_t stream = 1; stream < packed_test_streams; ++stream) {
            const llama_pos image_position = packed_test_positions[packed_test_offsets[stream]];
            llama_memory_seq_cp(
                    llama_get_memory(context.get()),
                    stream,
                    0,
                    image_position,
                    image_position + 1);
            if (!llama_memory_seq_rm(
                        llama_get_memory(context.get()),
                        stream,
                        image_position,
                        image_position + 1)) {
                throw std::runtime_error("D2F packed-lane sequence merge failed");
            }
        }

        const int32_t prompt_length = prefix_length - image_prefix_length;
        llama_batch prompt = llama_batch_init(prompt_length, 0, 1);
        for (int32_t i = 0; i < prompt_length; ++i) {
            const int32_t source = image_prefix_length + i;
            common_batch_add(prompt, tokens[source], positions[source], { 0 }, i == prompt_length - 1);
        }
        if (llama_decode(context.get(), prompt) != 0) {
            llama_batch_free(prompt);
            throw std::runtime_error("D2F packed-lane prompt decode failed");
        }
        llama_batch_free(prompt);

        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F packed-lane generation decode failed");
        }
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            logits_packed_lanes.insert(logits_packed_lanes.end(), row, row + n_vocab);
        }
        llama_batch_free(generation);
    }

    for (int32_t i = 0; i < generation_length; ++i) {
        const auto reference_begin = logits_packed_reference.begin() + static_cast<size_t>(i)*n_vocab;
        const auto packed_begin = logits_packed_lanes.begin() + static_cast<size_t>(i)*n_vocab;
        if (std::max_element(reference_begin, reference_begin + n_vocab) - reference_begin !=
            std::max_element(packed_begin, packed_begin + n_vocab) - packed_begin) {
            throw std::runtime_error("D2F packed-lane output token mismatch");
        }
    }

    const auto run_single_image_packed_parallel = [&](bool packed) {
        static constexpr int32_t lane_count = 3;
        static constexpr int32_t lane_size = 2;

        llama_context_params params = llama_context_default_params();
        params.n_ctx = 16;
        params.n_batch = image_prefix_length;
        params.n_ubatch = image_prefix_length;
        params.n_outputs_max = generation_length;
        params.n_seq_max = lane_count;
        params.kv_unified = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.no_perf = true;
        llama_context_ptr context(llama_init_from_model(model, params));
        if (!context) {
            throw std::runtime_error("failed to create single-image packed_parallel context");
        }
        llama_set_causal_attn(context.get(), false);
        llama_set_d2f_attention(
                context.get(),
                image_prefix_length,
                prefix_length,
                prompt_position,
                generation_position,
                block_size);
        if (packed) {
            llama_set_d2f_packed_prefill(
                    context.get(),
                    lane_count,
                    image_prefix_length,
                    lane_size);
        }

        const int32_t image_decodes = packed ? 1 : lane_count;
        for (int32_t decode_index = 0; decode_index < image_decodes; ++decode_index) {
            const int32_t offset = packed ? 0 : decode_index*lane_size;
            const int32_t length = packed ? image_prefix_length : lane_size;
            llama_batch image = llama_batch_init(length, packed_test_n_embd, 1);
            image.n_tokens = length;
            for (int32_t i = 0; i < length; ++i) {
                const int32_t source = offset + i;
                std::copy_n(
                        packed_test_image_embeddings.data() +
                                static_cast<size_t>(source)*packed_test_n_embd,
                        packed_test_n_embd,
                        image.embd + static_cast<size_t>(i)*packed_test_n_embd);
                image.pos[i] = packed_test_positions[source];
                image.n_seq_id[i] = 1;
                image.seq_id[i][0] = packed ? source/lane_size : decode_index;
                image.logits[i] = i == length - 1;
            }
            if (llama_decode(context.get(), image) != 0) {
                llama_batch_free(image);
                throw std::runtime_error("single-image packed_parallel image decode failed");
            }
            llama_batch_free(image);
        }

        if (packed) {
            llama_set_d2f_packed_prefill(context.get(), 0, 0, 0);
        }
        for (int32_t lane = 1; lane < lane_count; ++lane) {
            const int32_t offset = lane*lane_size;
            const auto first = packed_test_positions.begin() + offset;
            const auto last = first + lane_size;
            const llama_pos position_min = *std::min_element(first, last);
            const llama_pos position_max = *std::max_element(first, last);
            llama_memory_seq_cp(
                    llama_get_memory(context.get()),
                    lane,
                    0,
                    position_min,
                    position_max + 1);
            if (!llama_memory_seq_rm(
                        llama_get_memory(context.get()),
                        lane,
                        position_min,
                        position_max + 1)) {
                throw std::runtime_error("single-image packed_parallel lane merge failed");
            }
        }

        const int32_t prompt_length = prefix_length - image_prefix_length;
        llama_batch prompt = llama_batch_init(prompt_length, 0, 1);
        for (int32_t i = 0; i < prompt_length; ++i) {
            const int32_t source = image_prefix_length + i;
            common_batch_add(prompt, tokens[source], positions[source], { 0 }, i == prompt_length - 1);
        }
        if (llama_decode(context.get(), prompt) != 0) {
            llama_batch_free(prompt);
            throw std::runtime_error("single-image packed_parallel prompt decode failed");
        }
        llama_batch_free(prompt);

        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("single-image packed_parallel generation decode failed");
        }
        std::vector<float> result;
        result.reserve((generation_length + 1)*n_vocab);
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            result.insert(result.end(), row, row + n_vocab);
        }
        llama_batch_free(generation);

        llama_batch continuation = llama_batch_init(1, 0, 1);
        common_batch_add(continuation, tokens[0], generation_position + generation_length, { 0 }, true);
        if (llama_decode(context.get(), continuation) != 0) {
            llama_batch_free(continuation);
            throw std::runtime_error("single-image packed_parallel continuation decode failed");
        }
        const float * continuation_logits = llama_get_logits_ith(context.get(), 0);
        result.insert(result.end(), continuation_logits, continuation_logits + n_vocab);
        llama_batch_free(continuation);
        return result;
    };

    const std::vector<float> logits_packed_parallel_reference =
            run_single_image_packed_parallel(false);
    const std::vector<float> logits_packed_parallel =
            run_single_image_packed_parallel(true);
    for (int32_t i = 0; i < generation_length + 1; ++i) {
        const auto reference_begin =
                logits_packed_parallel_reference.begin() + static_cast<size_t>(i)*n_vocab;
        const auto packed_begin = logits_packed_parallel.begin() + static_cast<size_t>(i)*n_vocab;
        if (std::max_element(reference_begin, reference_begin + n_vocab) - reference_begin !=
            std::max_element(packed_begin, packed_begin + n_vocab) - packed_begin) {
            throw std::runtime_error("single-image packed_parallel continuation token mismatch");
        }
    }
    const double packed_parallel_nmse =
            nmse(logits_packed_parallel_reference, logits_packed_parallel);

    const auto run_quantized_packed_arm = [&](bool packed) {
        static constexpr int32_t packed_tokens = image_prefix_length;

        llama_context_params params = llama_context_default_params();
        params.n_ctx = 16;
        params.n_batch = packed_tokens;
        params.n_ubatch = packed_tokens;
        params.n_outputs_max = generation_length;
        params.n_seq_max = packed ? packed_test_streams : 1;
        params.kv_unified = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.type_k = GGML_TYPE_Q8_0;
        params.type_v = GGML_TYPE_Q8_0;
        params.no_perf = true;
        llama_context_ptr context(llama_init_from_model(model, params));
        if (!context) {
            throw std::runtime_error("failed to create D2F Q8_0 packed-lane context");
        }
        auto * cache = dynamic_cast<llama_kv_cache *>(llama_get_memory(context.get()));
        if (!cache || cache->type_k() != GGML_TYPE_Q8_0 || cache->type_v() != GGML_TYPE_Q8_0) {
            throw std::runtime_error("D2F Q8_0 packed-lane context did not retain quantized K/V");
        }
        llama_set_causal_attn(context.get(), false);
        llama_set_d2f_attention(
                context.get(),
                image_prefix_length,
                prefix_length,
                prompt_position,
                generation_position,
                block_size);
        if (packed) {
            llama_set_d2f_packed_prefill(context.get(), packed_test_streams, packed_tokens, 3);
        }

        const int32_t image_decodes = packed ? 1 : packed_test_streams;
        for (int32_t decode_index = 0; decode_index < image_decodes; ++decode_index) {
            const int32_t offset = packed ? 0 : packed_test_offsets[decode_index];
            const int32_t length = packed ? packed_tokens : packed_test_lengths[decode_index];
            llama_batch image = llama_batch_init(length, packed_test_n_embd, 1);
            image.n_tokens = length;
            for (int32_t i = 0; i < length; ++i) {
                const int32_t source = offset + i;
                std::copy_n(
                        packed_test_image_embeddings.data() + static_cast<size_t>(source)*packed_test_n_embd,
                        packed_test_n_embd,
                        image.embd + static_cast<size_t>(i)*packed_test_n_embd);
                image.pos[i] = packed_test_positions[source];
                image.n_seq_id[i] = 1;
                image.seq_id[i][0] = !packed
                        ? 0
                        : source < packed_test_offsets[1] ? 0 : source < packed_test_offsets[2] ? 1 : 2;
                image.logits[i] = i == length - 1;
            }
            if (llama_decode(context.get(), image) != 0) {
                llama_batch_free(image);
                throw std::runtime_error("D2F Q8_0 packed-lane image decode failed");
            }
            llama_batch_free(image);
        }

        if (packed) {
            llama_set_d2f_packed_prefill(context.get(), 0, 0, 0);
            for (int32_t stream = 1; stream < packed_test_streams; ++stream) {
                const llama_pos image_position = packed_test_positions[packed_test_offsets[stream]];
                llama_memory_seq_cp(
                        llama_get_memory(context.get()),
                        stream,
                        0,
                        image_position,
                        image_position + 1);
                if (!llama_memory_seq_rm(
                            llama_get_memory(context.get()),
                            stream,
                            image_position,
                            image_position + 1)) {
                    throw std::runtime_error("D2F Q8_0 packed-lane sequence merge failed");
                }
            }
        }

        const int32_t prompt_length = prefix_length - image_prefix_length;
        llama_batch prompt = llama_batch_init(prompt_length, 0, 1);
        for (int32_t i = 0; i < prompt_length; ++i) {
            const int32_t source = image_prefix_length + i;
            common_batch_add(prompt, tokens[source], positions[source], { 0 }, i == prompt_length - 1);
        }
        if (llama_decode(context.get(), prompt) != 0) {
            llama_batch_free(prompt);
            throw std::runtime_error("D2F Q8_0 packed-lane prompt decode failed");
        }
        llama_batch_free(prompt);

        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F Q8_0 packed-lane generation decode failed");
        }
        std::vector<float> result;
        result.reserve(generation_length*n_vocab);
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            result.insert(result.end(), row, row + n_vocab);
        }
        llama_batch_free(generation);
        return result;
    };

    const std::vector<float> logits_packed_q8_reference = run_quantized_packed_arm(false);
    const std::vector<float> logits_packed_q8_lanes = run_quantized_packed_arm(true);
    for (int32_t i = 0; i < generation_length; ++i) {
        const auto reference_begin = logits_packed_q8_reference.begin() + static_cast<size_t>(i)*n_vocab;
        const auto packed_begin = logits_packed_q8_lanes.begin() + static_cast<size_t>(i)*n_vocab;
        if (std::max_element(reference_begin, reference_begin + n_vocab) - reference_begin !=
            std::max_element(packed_begin, packed_begin + n_vocab) - packed_begin) {
            throw std::runtime_error("D2F Q8_0 packed-lane output token mismatch");
        }
    }
    const double packed_q8_nmse = nmse(logits_packed_q8_reference, logits_packed_q8_lanes);

    for (int32_t i = 0; i < generation_length; ++i) {
        const auto cached_begin = logits_cached.begin() + static_cast<size_t>(i)*n_vocab;
        const auto component_begin = logits_component_parallel.begin() + static_cast<size_t>(i)*n_vocab;
        const auto cached_best = std::max_element(cached_begin, cached_begin + n_vocab);
        const auto component_best = std::max_element(component_begin, component_begin + n_vocab);
        if (cached_best - cached_begin != component_best - component_begin) {
            throw std::runtime_error("D2F parallel component output token mismatch");
        }
    }

    double scheduler_nmse = 0.0;
    {
        llama_context_ptr exact_context = make_context();
        llama_batch exact_prefix = llama_batch_init(prefix_length, 0, 1);
        for (int32_t i = 0; i < prefix_length; ++i) {
            common_batch_add(
                    exact_prefix,
                    tokens[i],
                    positions[i],
                    { 0 },
                    i == prefix_length - 1);
        }
        if (llama_decode(exact_context.get(), exact_prefix) != 0) {
            llama_batch_free(exact_prefix);
            throw std::runtime_error("D2F scheduler exact prefix decode failed");
        }
        llama_batch_free(exact_prefix);

        llama_context_ptr component_parallel_context = make_context(image_prefix_length);
        llama_batch image = llama_batch_init(image_prefix_length, 0, 1);
        for (int32_t i = 0; i < image_prefix_length; ++i) {
            common_batch_add(
                    image,
                    tokens[i],
                    positions[i],
                    { 0 },
                    i == image_prefix_length - 1);
        }
        if (llama_decode(component_parallel_context.get(), image) != 0) {
            llama_batch_free(image);
            throw std::runtime_error("D2F scheduler parallel component image decode failed");
        }
        llama_batch_free(image);

        const int32_t prompt_length = prefix_length - image_prefix_length;
        llama_batch prompt = llama_batch_init(prompt_length, 0, 1);
        for (int32_t i = 0; i < prompt_length; ++i) {
            const int32_t source = image_prefix_length + i;
            common_batch_add(
                    prompt,
                    tokens[source],
                    positions[source],
                    { 0 },
                    i == prompt_length - 1);
        }
        if (llama_decode(component_parallel_context.get(), prompt) != 0) {
            llama_batch_free(prompt);
            throw std::runtime_error("D2F scheduler parallel component prompt decode failed");
        }
        llama_batch_free(prompt);

        diffusion_d2f_scheduler_params scheduler_params;
        scheduler_params.generation_length = generation_length;
        scheduler_params.block_length = block_size;
        scheduler_params.max_iterations = 32;
        scheduler_params.mask_token_id = 0;
        scheduler_params.initial_token_id = 1;
        scheduler_params.eos_token_id = -1;
        diffusion_d2f_scheduler exact_scheduler(scheduler_params);
        diffusion_d2f_scheduler component_parallel_scheduler(scheduler_params);

        auto decode_scheduler_step = [&](llama_context * context,
                                         const diffusion_d2f_scheduler & scheduler,
                                         const diffusion_d2f_step & step,
                                         const char * label) {
            if (!llama_memory_seq_rm(
                        llama_get_memory(context),
                        0,
                        generation_position,
                        -1)) {
                throw std::runtime_error(std::string(label) + " dynamic range removal failed");
            }
            llama_batch generation = llama_batch_init(step.active_end, 0, 1);
            for (int32_t i = 0; i < step.active_end; ++i) {
                common_batch_add(
                        generation,
                        scheduler.tokens()[i],
                        generation_position + i,
                        { 0 },
                        true);
            }
            if (llama_decode(context, generation) != 0) {
                llama_batch_free(generation);
                throw std::runtime_error(std::string(label) + " generation decode failed");
            }
            const float * logits = llama_get_logits(context);
            if (!logits) {
                llama_batch_free(generation);
                throw std::runtime_error(std::string(label) + " returned no logits");
            }
            std::vector<float> result(
                    logits,
                    logits + static_cast<size_t>(step.active_end)*n_vocab);
            llama_batch_free(generation);
            return result;
        };

        int32_t scheduler_iterations = 0;
        while (!exact_scheduler.done() || !component_parallel_scheduler.done()) {
            const diffusion_d2f_step exact_step = exact_scheduler.prepare_step();
            const diffusion_d2f_step component_step = component_parallel_scheduler.prepare_step();
            if (exact_step.iteration != component_step.iteration ||
                exact_step.active_start != component_step.active_start ||
                exact_step.active_end != component_step.active_end ||
                exact_step.blocks_added != component_step.blocks_added ||
                exact_step.blocks_completed != component_step.blocks_completed ||
                exact_step.done != component_step.done) {
                throw std::runtime_error("D2F parallel component scheduler step mismatch");
            }
            if (exact_step.done) {
                break;
            }

            const std::vector<float> exact_logits = decode_scheduler_step(
                    exact_context.get(), exact_scheduler, exact_step, "exact scheduler");
            const std::vector<float> component_logits = decode_scheduler_step(
                    component_parallel_context.get(), component_parallel_scheduler, component_step,
                    "parallel component scheduler");
            scheduler_nmse = std::max(
                    scheduler_nmse,
                    nmse(exact_logits, component_logits));

            const std::vector<diffusion_d2f_candidate> exact_candidates =
                    diffusion_d2f_argmax_candidates(
                            exact_logits.data(),
                            exact_step.active_end,
                            n_vocab,
                            0,
                            0,
                            generation_length,
                            false);
            const std::vector<diffusion_d2f_candidate> component_candidates =
                    diffusion_d2f_argmax_candidates(
                            component_logits.data(),
                            component_step.active_end,
                            n_vocab,
                            0,
                            0,
                            generation_length,
                            false);
            const std::vector<diffusion_d2f_update> exact_updates =
                    exact_scheduler.apply_candidates(exact_candidates);
            const std::vector<diffusion_d2f_update> component_updates =
                    component_parallel_scheduler.apply_candidates(component_candidates);
            if (exact_updates.size() != component_updates.size()) {
                throw std::runtime_error("D2F parallel component scheduler update count mismatch");
            }
            for (size_t i = 0; i < exact_updates.size(); ++i) {
                if (exact_updates[i].position != component_updates[i].position ||
                    exact_updates[i].token_id != component_updates[i].token_id) {
                    throw std::runtime_error("D2F parallel component scheduler update mismatch");
                }
            }
            if (exact_scheduler.tokens() != component_parallel_scheduler.tokens()) {
                throw std::runtime_error("D2F parallel component scheduler token trace mismatch");
            }
            if (++scheduler_iterations > scheduler_params.max_iterations) {
                throw std::runtime_error("D2F parallel component scheduler did not finish");
            }
        }
        if (exact_scheduler.done() != component_parallel_scheduler.done() ||
            exact_scheduler.output_length() != component_parallel_scheduler.output_length() ||
            exact_scheduler.tokens() != component_parallel_scheduler.tokens()) {
            throw std::runtime_error("D2F parallel component scheduler output mismatch");
        }
    }

    {
        llama_context_ptr context = make_context(4);
        const int32_t n_embd = llama_model_n_embd_inp(model);
        for (int32_t offset = 0; offset < image_prefix_length; offset += 2) {
            const int32_t length = std::min(2, image_prefix_length - offset);
            llama_batch image = llama_batch_init(length, n_embd, 1);
            for (int32_t i = 0; i < length; ++i) {
                const int32_t source = offset + i;
                for (int32_t j = 0; j < n_embd; ++j) {
                    image.embd[static_cast<size_t>(i)*n_embd + j] =
                            static_cast<float>((tokens[source] + j) % 17) / 17.0f;
                }
                image.pos[i] = positions[source];
                image.n_seq_id[i] = 1;
                image.seq_id[i][0] = 0;
                image.logits[i] = i == length - 1;
            }
            image.n_tokens = length;
            if (llama_decode(context.get(), image) != 0) {
                llama_batch_free(image);
                throw std::runtime_error("D2F packed image decode failed");
            }
            llama_batch_free(image);
        }

        const int32_t prompt_length = prefix_length - image_prefix_length;
        llama_batch prompt = llama_batch_init(prompt_length, 0, 1);
        for (int32_t i = 0; i < prompt_length; ++i) {
            const int32_t source = image_prefix_length + i;
            common_batch_add(prompt, tokens[source], positions[source], { 0 }, i == prompt_length - 1);
        }
        if (llama_decode(context.get(), prompt) != 0) {
            llama_batch_free(prompt);
            throw std::runtime_error("D2F packed image prompt decode failed");
        }
        llama_batch_free(prompt);

        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F packed image generation decode failed");
        }
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            for (uint32_t token = 0; token < n_vocab; ++token) {
                if (!std::isfinite(row[token])) {
                    llama_batch_free(generation);
                    throw std::runtime_error("D2F packed image logits are not finite");
                }
            }
        }
        llama_batch_free(generation);
    }

    std::vector<uint8_t> prefix_state;
    {
        llama_context_ptr context = make_context();
        llama_batch prefix = llama_batch_init(prefix_length, 0, 1);
        for (int32_t i = 0; i < prefix_length; ++i) {
            common_batch_add(prefix, tokens[i], positions[i], { 0 }, i == prefix_length - 1);
        }
        if (llama_decode(context.get(), prefix) != 0) {
            llama_batch_free(prefix);
            throw std::runtime_error("D2F transferred prefix decode failed");
        }
        llama_batch_free(prefix);

        prefix_state.resize(llama_state_seq_get_size(context.get(), 0));
        const size_t copied = llama_state_seq_get_data(
                context.get(), prefix_state.data(), prefix_state.size(), 0);
        if (copied == 0 || copied != prefix_state.size()) {
            throw std::runtime_error("D2F prefix state export failed");
        }
    }

    std::vector<float> logits_transferred;
    logits_transferred.reserve(generation_length*n_vocab);
    {
        llama_context_ptr context = make_context();
        const size_t restored = llama_state_seq_set_data(
                context.get(), prefix_state.data(), prefix_state.size(), 0);
        if (restored == 0 || restored != prefix_state.size()) {
            throw std::runtime_error("D2F prefix state import failed");
        }

        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F transferred generation decode failed");
        }
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context.get(), i);
            logits_transferred.insert(logits_transferred.end(), row, row + n_vocab);
        }
        llama_batch_free(generation);
    }

    const double full_cache_nmse = nmse(logits_full, logits_cached);
    const double component_parallel_nmse = nmse(logits_cached, logits_component_parallel);
    const double packed_lanes_nmse = nmse(logits_packed_reference, logits_packed_lanes);
    const double transferred_nmse = nmse(logits_cached, logits_transferred);
    std::fprintf(
            stderr,
            "D2F prefix equivalence: full_cache_nmse=%.9e component_parallel_nmse=%.9e "
            "packed_lanes_nmse=%.9e packed_parallel_nmse=%.9e packed_q8_nmse=%.9e scheduler_nmse=%.9e "
            "transferred_nmse=%.9e output_tokens=match\n",
            full_cache_nmse,
            component_parallel_nmse,
            packed_lanes_nmse,
            packed_parallel_nmse,
            packed_q8_nmse,
            scheduler_nmse,
            transferred_nmse);
    return std::max({
            full_cache_nmse,
            component_parallel_nmse,
            packed_lanes_nmse,
            packed_parallel_nmse,
            packed_q8_nmse,
            transferred_nmse,
            scheduler_nmse,
    });
}

static void test_d2f_kv_head_compaction(llama_model * model, const std::vector<llama_token> & tokens) {
    static const std::vector<llama_pos> positions = { 0, 0, 0, 0, 1, 2, 3, 4, 5, 6 };
    static constexpr int32_t image_prefix_length = 4;
    static constexpr int32_t prefix_length = 6;
    static constexpr int32_t prompt_position = 1;
    static constexpr int32_t generation_position = 3;
    static constexpr int32_t block_size = 2;
    static constexpr int32_t generation_length = 4;
    static constexpr uint32_t compact_length = 4;

    GGML_ASSERT(tokens.size() >= positions.size());

    auto make_context = [model]() {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 16;
        params.n_batch = 16;
        params.n_ubatch = 16;
        params.n_outputs_max = generation_length;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.no_perf = true;

        llama_context_ptr context(llama_init_from_model(model, params));
        if (!context) {
            throw std::runtime_error("failed to create D2F KV compaction test context");
        }
        llama_set_causal_attn(context.get(), false);
        llama_set_d2f_attention(
                context.get(),
                image_prefix_length,
                prefix_length,
                prompt_position,
                generation_position,
                block_size);
        return context;
    };

    auto decode_prefix = [&](llama_context * context) {
        llama_batch prefix = llama_batch_init(prefix_length, 0, 1);
        for (int32_t i = 0; i < prefix_length; ++i) {
            common_batch_add(prefix, tokens[i], positions[i], { 0 }, i == prefix_length - 1);
        }
        if (llama_decode(context, prefix) != 0) {
            llama_batch_free(prefix);
            throw std::runtime_error("D2F KV compaction prefix decode failed");
        }
        llama_batch_free(prefix);
    };

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto decode_generation = [&](llama_context * context) {
        llama_batch generation = llama_batch_init(generation_length, 0, 1);
        for (int32_t i = 0; i < generation_length; ++i) {
            const int32_t source = prefix_length + i;
            common_batch_add(generation, tokens[source], positions[source], { 0 }, true);
        }
        if (llama_decode(context, generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F KV compaction continuation decode failed");
        }

        std::vector<float> result;
        result.reserve(generation_length*n_vocab);
        for (int32_t i = 0; i < generation_length; ++i) {
            const float * row = llama_get_logits_ith(context, i);
            result.insert(result.end(), row, row + n_vocab);
        }
        llama_batch_free(generation);
        return result;
    };

    auto get_cache = [](llama_context * context) {
        auto * cache = dynamic_cast<llama_kv_cache *>(llama_get_memory(context));
        if (!cache) {
            throw std::runtime_error("D2F KV compaction context does not use a KV cache");
        }
        if (cache->type_k() != GGML_TYPE_F16 || cache->type_v() != GGML_TYPE_F16) {
            throw std::runtime_error("D2F KV compaction test requires f16 K and V caches");
        }
        return cache;
    };

    llama_context_ptr reference = make_context();
    decode_prefix(reference.get());
    const std::vector<float> reference_logits = decode_generation(reference.get());

    double identity_nmse = 0.0;
    {
        llama_context_ptr context = make_context();
        decode_prefix(context.get());
        llama_kv_cache * cache = get_cache(context.get());
        const std::vector<uint32_t> layer_ids = cache->get_layer_ids();
        const uint32_t n_head_kv = llama_model_n_head_kv(model);
        std::vector<uint32_t> identity(layer_ids.size()*n_head_kv*prefix_length);
        for (size_t il = 0; il < layer_ids.size(); ++il) {
            for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
                const size_t offset = (il*n_head_kv + ih)*prefix_length;
                for (uint32_t i = 0; i < prefix_length; ++i) {
                    identity[offset + i] = i;
                }
            }
        }
        const std::vector<llama_pos> dst_positions(positions.begin(), positions.begin() + prefix_length);
        if (!llama_kv_cache_compact_heads(
                    context.get(),
                    0,
                    prefix_length,
                    prefix_length,
                    identity.data(),
                    identity.size(),
                    dst_positions.data())) {
            throw std::runtime_error("D2F identity KV head compaction failed");
        }
        const std::vector<float> identity_logits = decode_generation(context.get());
        identity_nmse = nmse(reference_logits, identity_logits);
        if (identity_nmse > 1e-12) {
            throw std::runtime_error("D2F identity KV head compaction changed logits");
        }
    }

    size_t state_size_before = 0;
    size_t state_size_after = 0;
    {
        llama_context_ptr context = make_context();
        decode_prefix(context.get());
        llama_kv_cache * cache = get_cache(context.get());
        const std::vector<uint32_t> layer_ids = cache->get_layer_ids();
        const uint32_t n_head_kv = llama_model_n_head_kv(model);
        if (layer_ids.size() != 2 || n_head_kv != 2) {
            throw std::runtime_error("D2F KV compaction fixture needs two layers and two KV heads");
        }

        struct k_snapshot {
            size_t head_bytes;
            size_t row_bytes;
            std::vector<uint8_t> data;
        };
        std::vector<k_snapshot> before;
        before.reserve(layer_ids.size());
        for (uint32_t layer_id : layer_ids) {
            ggml_tensor * k = cache->get_k_storage(layer_id);
            if (!k || k->type != GGML_TYPE_F16 || k->ne[0] % n_head_kv != 0) {
                throw std::runtime_error("D2F KV compaction fixture has an unexpected K layout");
            }
            const int64_t n_embd_head = k->ne[0]/n_head_kv;
            k_snapshot snapshot {
                ggml_row_size(k->type, n_embd_head),
                ggml_row_size(k->type, k->ne[0]),
                {},
            };
            snapshot.data.resize(prefix_length*snapshot.row_bytes);
            ggml_backend_tensor_get(k, snapshot.data.data(), 0, snapshot.data.size());
            before.push_back(std::move(snapshot));
        }

        static constexpr uint32_t image_choices[4][2] = {
            { 0, 1 },
            { 0, 2 },
            { 0, 3 },
            { 1, 2 },
        };
        std::vector<uint32_t> sources(layer_ids.size()*n_head_kv*compact_length);
        for (size_t il = 0; il < layer_ids.size(); ++il) {
            for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
                const size_t offset = (il*n_head_kv + ih)*compact_length;
                const uint32_t choice = (il*n_head_kv + ih) % 4;
                sources[offset + 0] = image_choices[choice][0];
                sources[offset + 1] = image_choices[choice][1];
                sources[offset + 2] = 4;
                sources[offset + 3] = 5;
            }
        }
        const size_t n_groups = layer_ids.size()*n_head_kv;
        for (size_t lhs = 0; lhs < n_groups; ++lhs) {
            for (size_t rhs = lhs + 1; rhs < n_groups; ++rhs) {
                if (sources[lhs*compact_length] == sources[rhs*compact_length] &&
                    sources[lhs*compact_length + 1] == sources[rhs*compact_length + 1]) {
                    throw std::runtime_error("D2F KV compaction fixture repeated a per-layer per-head source map");
                }
            }
        }

        const std::vector<llama_pos> dst_positions = { 0, 0, 1, 2 };
        state_size_before = llama_state_seq_get_size(context.get(), 0);
        if (!llama_kv_cache_compact_heads(
                    context.get(),
                    0,
                    prefix_length,
                    compact_length,
                    sources.data(),
                    sources.size(),
                    dst_positions.data())) {
            throw std::runtime_error("D2F per-layer per-head KV compaction failed");
        }
        state_size_after = llama_state_seq_get_size(context.get(), 0);
        if (state_size_after >= state_size_before) {
            throw std::runtime_error("D2F KV compaction did not reduce serialized cache state");
        }
        if (llama_memory_seq_pos_min(llama_get_memory(context.get()), 0) != 0 ||
            llama_memory_seq_pos_max(llama_get_memory(context.get()), 0) != 2) {
            throw std::runtime_error("D2F KV compaction produced incorrect cell positions");
        }

        for (size_t il = 0; il < layer_ids.size(); ++il) {
            ggml_tensor * k = cache->get_k_storage(layer_ids[il]);
            std::vector<uint8_t> after(compact_length*before[il].row_bytes);
            ggml_backend_tensor_get(k, after.data(), 0, after.size());
            for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
                const size_t source_offset = (il*n_head_kv + ih)*compact_length;
                for (uint32_t idst = 0; idst < compact_length; ++idst) {
                    const uint32_t isrc = sources[source_offset + idst];
                    const uint8_t * expected = before[il].data.data() + isrc*before[il].row_bytes + ih*before[il].head_bytes;
                    const uint8_t * actual = after.data() + idst*before[il].row_bytes + ih*before[il].head_bytes;
                    if (memcmp(expected, actual, before[il].head_bytes) != 0) {
                        throw std::runtime_error("D2F KV compaction copied an incorrect per-head K row");
                    }
                }
            }
        }

        llama_set_d2f_attention(
                context.get(),
                2,
                compact_length,
                prompt_position,
                generation_position,
                block_size);
        const std::vector<float> continued_logits = decode_generation(context.get());
        for (float value : continued_logits) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("D2F KV compaction continuation produced non-finite logits");
            }
        }
        if (llama_memory_seq_pos_max(llama_get_memory(context.get()), 0) != positions.back()) {
            throw std::runtime_error("D2F KV compaction cache metadata did not advance after continuation");
        }
    }

    std::fprintf(
            stderr,
            "D2F KV head compaction: type=f16 layers=%d heads=%d identity_nmse=%.9e state=%zu->%zu continuation=finite\n",
            llama_model_n_layer(model),
            llama_model_n_head_kv(model),
            identity_nmse,
            state_size_before,
            state_size_after);
}

struct d2f_decode_iteration_trace {
    diffusion_d2f_step step;
    std::vector<int32_t> logit_positions;
    std::vector<float> logits;
    std::vector<diffusion_d2f_candidate> candidates;
    std::vector<diffusion_d2f_update> updates;
    std::vector<int32_t> tokens;
};

struct d2f_decode_variant_trace {
    std::vector<d2f_decode_iteration_trace> iterations;
    lladao::detail::d2f_decode_counters counters;
};

static d2f_decode_variant_trace run_d2f_decode_variant(
        llama_model * model,
        const std::vector<llama_token> & prefix_tokens,
        bool prefix_cache,
        bool generation_block_cache,
        bool sparse_logits,
        const std::vector<uint8_t> * restored_prefix_state = nullptr) {
    static const std::vector<llama_pos> prefix_positions = { 0, 0, 0, 1, 2 };
    static constexpr int32_t image_prefix_length = 3;
    static constexpr int32_t prefix_length = 5;
    static constexpr int32_t prompt_position = 1;
    static constexpr int32_t generation_position = 3;
    static constexpr int32_t block_size = 2;
    static constexpr int32_t generation_length = 4;
    static constexpr int32_t mask_token = 0;

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 16;
    context_params.n_batch = 16;
    context_params.n_ubatch = 16;
    context_params.n_outputs_max = generation_length;
    context_params.no_perf = true;
    llama_context_ptr context(llama_init_from_model(model, context_params));
    if (!context) {
        throw std::runtime_error("failed to create D2F generation optimization test context");
    }
    llama_set_causal_attn(context.get(), false);
    llama_set_d2f_attention(
            context.get(),
            image_prefix_length,
            prefix_length,
            prompt_position,
            generation_position,
            block_size);

    if (restored_prefix_state && !prefix_cache) {
        throw std::invalid_argument("D2F restored prefix state requires prefix caching");
    }
    if (restored_prefix_state) {
        const size_t restored = llama_state_seq_set_data(
                context.get(),
                restored_prefix_state->data(),
                restored_prefix_state->size(),
                0);
        if (restored == 0 || restored != restored_prefix_state->size()) {
            throw std::runtime_error("D2F generation optimization prefix restore failed");
        }
    } else if (prefix_cache) {
        llama_batch prefix = llama_batch_init(prefix_length, 0, 1);
        for (int32_t position = 0; position < prefix_length; ++position) {
            common_batch_add(
                    prefix,
                    prefix_tokens[position],
                    prefix_positions[position],
                    { 0 },
                    position == prefix_length - 1);
        }
        if (llama_decode(context.get(), prefix) != 0) {
            llama_batch_free(prefix);
            throw std::runtime_error("D2F generation optimization prefix decode failed");
        }
        llama_batch_free(prefix);
    }

    diffusion_d2f_scheduler_params scheduler_params;
    scheduler_params.generation_length = generation_length;
    scheduler_params.block_length = block_size;
    scheduler_params.max_iterations = 32;
    scheduler_params.mask_token_id = mask_token;
    scheduler_params.initial_token_id = 1;
    scheduler_params.eos_token_id = -1;
    diffusion_d2f_scheduler scheduler(scheduler_params);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    d2f_decode_variant_trace result;
    int32_t cached_complete_end = 0;
    while (!scheduler.done()) {
        const diffusion_d2f_step step = scheduler.prepare_step();
        if (step.done) {
            break;
        }
        const bool resolved_generation_block_cache = prefix_cache && generation_block_cache;
        const lladao::detail::d2f_generation_decode_plan plan =
                lladao::detail::plan_d2f_generation_decode(
                        scheduler.tokens(),
                        mask_token,
                        step.active_start,
                        step.active_end,
                        block_size,
                        cached_complete_end,
                        resolved_generation_block_cache,
                        sparse_logits);
        if (prefix_cache) {
            if (!llama_memory_seq_rm(
                        llama_get_memory(context.get()),
                        0,
                        generation_position + plan.first_position,
                        -1)) {
                throw std::runtime_error("D2F generation optimization cache removal failed");
            }
        } else {
            llama_memory_clear(llama_get_memory(context.get()), true);
        }

        const int32_t batch_prefix_rows = prefix_cache ? 0 : prefix_length;
        llama_batch generation = llama_batch_init(batch_prefix_rows + plan.input_rows(), 0, 1);
        if (!prefix_cache) {
            for (int32_t position = 0; position < prefix_length; ++position) {
                common_batch_add(
                        generation,
                        prefix_tokens[position],
                        prefix_positions[position],
                        { 0 },
                        false);
            }
        }
        for (int32_t position = plan.first_position; position < plan.active_end; ++position) {
            common_batch_add(
                    generation,
                    scheduler.tokens()[position],
                    generation_position + position,
                    { 0 },
                    plan.requests_logits(position));
        }
        if (llama_decode(context.get(), generation) != 0) {
            llama_batch_free(generation);
            throw std::runtime_error("D2F generation optimization decode failed");
        }

        d2f_decode_iteration_trace iteration;
        iteration.step = step;
        iteration.candidates.resize(generation_length);
        for (int32_t position = step.active_start; position < step.active_end; ++position) {
            if (scheduler.tokens()[position] != mask_token) {
                continue;
            }
            const int32_t batch_index = plan.batch_index(position, batch_prefix_rows);
            const float * row = llama_get_logits_ith(context.get(), batch_index);
            if (!row) {
                llama_batch_free(generation);
                throw std::runtime_error("D2F generation optimization returned no sparse logits row");
            }
            iteration.logit_positions.push_back(position);
            iteration.logits.insert(iteration.logits.end(), row, row + n_vocab);
            iteration.candidates[position] = diffusion_d2f_argmax_candidate(row, n_vocab);
        }
        llama_batch_free(generation);

        iteration.updates = scheduler.apply_candidates(iteration.candidates);
        iteration.tokens = scheduler.tokens();
        result.iterations.push_back(std::move(iteration));
        result.counters.add(plan, batch_prefix_rows);
        cached_complete_end = plan.next_cached_complete_end;
    }
    if (!scheduler.done()) {
        throw std::runtime_error("D2F generation optimization trace did not complete");
    }
    return result;
}

static void require_d2f_decode_trace_exact(
        const d2f_decode_variant_trace & expected,
        const d2f_decode_variant_trace & actual,
        const char * label) {
    if (actual.iterations.size() != expected.iterations.size()) {
        throw std::runtime_error(std::string(label) + " iteration count mismatch");
    }
    for (size_t index = 0; index < expected.iterations.size(); ++index) {
        const d2f_decode_iteration_trace & left = expected.iterations[index];
        const d2f_decode_iteration_trace & right = actual.iterations[index];
        if (left.step.iteration != right.step.iteration ||
            left.step.active_start != right.step.active_start ||
            left.step.active_end != right.step.active_end ||
            left.step.blocks_added != right.step.blocks_added ||
            left.step.blocks_completed != right.step.blocks_completed ||
            left.step.done != right.step.done ||
            left.logit_positions != right.logit_positions ||
            left.logits != right.logits ||
            left.tokens != right.tokens ||
            left.candidates.size() != right.candidates.size() ||
            left.updates.size() != right.updates.size()) {
            throw std::runtime_error(
                    std::string(label) + " iteration " + std::to_string(index + 1) +
                    " logits or scheduler trace mismatch");
        }
        for (size_t position = 0; position < left.candidates.size(); ++position) {
            const diffusion_d2f_candidate & a = left.candidates[position];
            const diffusion_d2f_candidate & b = right.candidates[position];
            if (a.token_id != b.token_id || a.confidence != b.confidence || a.valid != b.valid) {
                throw std::runtime_error(std::string(label) + " candidate trace mismatch");
            }
        }
        for (size_t update = 0; update < left.updates.size(); ++update) {
            const diffusion_d2f_update & a = left.updates[update];
            const diffusion_d2f_update & b = right.updates[update];
            if (a.position != b.position || a.token_id != b.token_id || a.confidence != b.confidence) {
                throw std::runtime_error(std::string(label) + " update trace mismatch");
            }
        }
    }
}

static std::vector<uint8_t> make_d2f_decode_prefix_state(
        llama_model * model,
        const std::vector<llama_token> & prefix_tokens) {
    static const std::vector<llama_pos> prefix_positions = { 0, 0, 0, 1, 2 };
    static constexpr int32_t image_prefix_length = 3;
    static constexpr int32_t prefix_length = 5;
    static constexpr int32_t prompt_position = 1;
    static constexpr int32_t generation_position = 3;
    static constexpr int32_t block_size = 2;
    static constexpr int32_t generation_length = 4;

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 16;
    context_params.n_batch = 16;
    context_params.n_ubatch = 16;
    context_params.n_outputs_max = generation_length;
    context_params.no_perf = true;
    llama_context_ptr context(llama_init_from_model(model, context_params));
    if (!context) {
        throw std::runtime_error("failed to create D2F PD exact A/B context");
    }
    llama_set_causal_attn(context.get(), false);
    llama_set_d2f_attention(
            context.get(),
            image_prefix_length,
            prefix_length,
            prompt_position,
            generation_position,
            block_size);

    llama_batch prefix = llama_batch_init(prefix_length, 0, 1);
    for (int32_t position = 0; position < prefix_length; ++position) {
        common_batch_add(
                prefix,
                prefix_tokens[position],
                prefix_positions[position],
                { 0 },
                position == prefix_length - 1);
    }
    if (llama_decode(context.get(), prefix) != 0) {
        llama_batch_free(prefix);
        throw std::runtime_error("D2F PD exact A/B prefix decode failed");
    }
    llama_batch_free(prefix);

    std::vector<uint8_t> state(llama_state_seq_get_size(context.get(), 0));
    const size_t copied = llama_state_seq_get_data(
            context.get(), state.data(), state.size(), 0);
    if (copied == 0 || copied != state.size()) {
        throw std::runtime_error("D2F PD exact A/B prefix export failed");
    }
    return state;
}

static void test_d2f_generation_cache_sparse(
        llama_model * model,
        const std::vector<llama_token> & tokens) {
    const d2f_decode_variant_trace full_dense =
            run_d2f_decode_variant(model, tokens, true, false, false);
    const d2f_decode_variant_trace cached_dense =
            run_d2f_decode_variant(model, tokens, true, true, false);
    const d2f_decode_variant_trace full_sparse =
            run_d2f_decode_variant(model, tokens, true, false, true);
    const d2f_decode_variant_trace cached_sparse =
            run_d2f_decode_variant(model, tokens, true, true, true);

    require_d2f_decode_trace_exact(full_dense, cached_dense, "generation block cache");
    require_d2f_decode_trace_exact(full_dense, full_sparse, "sparse generation logits");
    require_d2f_decode_trace_exact(full_dense, cached_sparse, "combined D2F optimizations");

    const d2f_decode_variant_trace full_sequence_dense =
            run_d2f_decode_variant(model, tokens, false, false, false);
    const d2f_decode_variant_trace full_sequence_sparse =
            run_d2f_decode_variant(model, tokens, false, false, true);
    require_d2f_decode_trace_exact(
            full_sequence_dense,
            full_sequence_sparse,
            "no-prefix-cache sparse generation logits");

    const std::vector<uint8_t> prefix_state = make_d2f_decode_prefix_state(model, tokens);
    const d2f_decode_variant_trace restored_full_dense =
            run_d2f_decode_variant(model, tokens, true, false, false, &prefix_state);
    const d2f_decode_variant_trace restored_cached_sparse =
            run_d2f_decode_variant(model, tokens, true, true, true, &prefix_state);
    require_d2f_decode_trace_exact(full_dense, restored_full_dense, "PD restored prefix");
    require_d2f_decode_trace_exact(
            restored_full_dense,
            restored_cached_sparse,
            "PD restored D2F optimizations");

    if (full_dense.counters.input_rows != full_dense.counters.logit_rows ||
        cached_dense.counters.input_rows != cached_dense.counters.logit_rows ||
        cached_dense.counters.input_rows >= full_dense.counters.input_rows ||
        full_sparse.counters.input_rows != full_dense.counters.input_rows ||
        full_sparse.counters.logit_rows >= full_dense.counters.logit_rows ||
        cached_sparse.counters.input_rows != cached_dense.counters.input_rows ||
        cached_sparse.counters.logit_rows != full_sparse.counters.logit_rows ||
        cached_sparse.counters.rebuild_rows == 0 ||
        cached_sparse.counters.reused_input_rows !=
                full_dense.counters.input_rows - cached_sparse.counters.input_rows ||
        full_sequence_dense.counters.input_rows != full_sequence_sparse.counters.input_rows ||
        full_sequence_sparse.counters.logit_rows >= full_sequence_dense.counters.logit_rows ||
        restored_full_dense.counters.input_rows != full_dense.counters.input_rows ||
        restored_cached_sparse.counters.input_rows != cached_sparse.counters.input_rows ||
        restored_cached_sparse.counters.logit_rows != cached_sparse.counters.logit_rows) {
        throw std::runtime_error("D2F generation optimization row counters did not show the expected reductions");
    }

    std::fprintf(
            stderr,
            "D2F generation exact A/B: iterations=%zu full_dense_input=%" PRIu64
            " cached_input=%" PRIu64 " full_dense_logits=%" PRIu64
            " sparse_logits=%" PRIu64 " rebuild_rows=%" PRIu64
            " reused_input_rows=%" PRIu64
            " full_sequence_input=%" PRIu64 " full_sequence_sparse_logits=%" PRIu64
            " pd_restore=exact logits_candidates_updates=exact\n",
            full_dense.iterations.size(),
            full_dense.counters.input_rows,
            cached_sparse.counters.input_rows,
            full_dense.counters.logit_rows,
            cached_sparse.counters.logit_rows,
            cached_sparse.counters.rebuild_rows,
            cached_sparse.counters.reused_input_rows,
            full_sequence_sparse.counters.input_rows,
            full_sequence_sparse.counters.logit_rows);
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    if (arch == LLM_ARCH_DEEPSEEK4) {
        return false;
    }

    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                        if (arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE) {
                            const double cache_nmse = test_d2f_prefix_cache(model_and_ctx_cpu.first.get(), tokens);
                            if (cache_nmse > 1e-5) {
                                throw std::runtime_error(
                                        "D2F prefix-cache logits mismatch: NMSE=" + std::to_string(cache_nmse));
                            }
                            test_d2f_kv_head_compaction(model_and_ctx_cpu.first.get(), tokens);
                            test_d2f_generation_cache_sparse(model_and_ctx_cpu.first.get(), tokens);
                        }
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting
    common_init();
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;
    std::string out;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!out.empty()) {
            return save_models(arch, seed, log_level, out);
        }
        return test_backends(arch, seed, log_level);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
