#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lladao::detail {

enum class d2f_qk_capture_phase : int32_t {
    inactive,
    image,
    prompt,
};

enum class d2f_qk_tensor_kind : int32_t {
    query,
    key,
};

struct d2f_qk_tensor_name {
    d2f_qk_tensor_kind kind = d2f_qk_tensor_kind::query;
    int32_t layer_index = -1;
};

struct d2f_qk_capture_config {
    bool enabled = false;
    int32_t total_layers = 0;
    int32_t last_n_layers = 0;
    int32_t query_heads = 0;
    int32_t kv_heads = 0;
    int32_t head_dim = 0;
};

struct d2f_qk_tensor_shape {
    int32_t tokens = 0;
    int32_t heads = 0;
    int32_t head_dim = 0;
};

struct d2f_qk_host_tensor {
    d2f_qk_capture_phase phase = d2f_qk_capture_phase::inactive;
    d2f_qk_tensor_kind kind = d2f_qk_tensor_kind::query;
    int32_t layer_index = -1;
    d2f_qk_tensor_shape shape;
    std::vector<float> values;

    float at(int32_t token, int32_t head, int32_t dim) const;
};

inline const char * d2f_qk_capture_phase_name(d2f_qk_capture_phase phase) {
    switch (phase) {
        case d2f_qk_capture_phase::inactive: return "inactive";
        case d2f_qk_capture_phase::image:    return "image";
        case d2f_qk_capture_phase::prompt:   return "prompt";
    }
    return "invalid";
}

inline const char * d2f_qk_tensor_kind_name(d2f_qk_tensor_kind kind) {
    switch (kind) {
        case d2f_qk_tensor_kind::query: return "query";
        case d2f_qk_tensor_kind::key:   return "key";
    }
    return "invalid";
}

inline std::optional<d2f_qk_tensor_name> parse_d2f_post_rope_qk_name(std::string_view name) {
    d2f_qk_tensor_kind kind;
    std::string_view suffix;
    if (name.size() > 5 && name.substr(0, 5) == "Qcur-") {
        kind = d2f_qk_tensor_kind::query;
        suffix = name.substr(5);
    } else if (name.size() > 5 && name.substr(0, 5) == "Kcur-") {
        kind = d2f_qk_tensor_kind::key;
        suffix = name.substr(5);
    } else {
        return std::nullopt;
    }

    if (suffix.empty() || (suffix.size() > 1 && suffix.front() == '0')) {
        return std::nullopt;
    }
    int32_t layer_index = 0;
    for (char digit : suffix) {
        if (digit < '0' || digit > '9') {
            return std::nullopt;
        }
        const int32_t value = digit - '0';
        if (layer_index > (std::numeric_limits<int32_t>::max() - value) / 10) {
            return std::nullopt;
        }
        layer_index = layer_index * 10 + value;
    }
    return d2f_qk_tensor_name { kind, layer_index };
}

inline void validate_d2f_qk_capture_config(const d2f_qk_capture_config & config) {
    if (!config.enabled) {
        return;
    }
    if (config.total_layers <= 0 || config.last_n_layers <= 0 ||
        config.last_n_layers > config.total_layers || config.query_heads <= 0 ||
        config.kv_heads <= 0 || config.head_dim <= 0) {
        throw std::invalid_argument("invalid post-RoPE Q/K capture configuration");
    }
}

inline bool d2f_qk_layer_selected(const d2f_qk_capture_config & config, int32_t layer_index) {
    validate_d2f_qk_capture_config(config);
    return config.enabled && layer_index >= config.total_layers - config.last_n_layers &&
           layer_index < config.total_layers;
}

inline size_t d2f_qk_element_count(const d2f_qk_tensor_shape & shape) {
    if (shape.tokens <= 0 || shape.heads <= 0 || shape.head_dim <= 0) {
        throw std::invalid_argument("invalid post-RoPE Q/K tensor shape");
    }
    size_t result = static_cast<size_t>(shape.tokens);
    for (int32_t factor : { shape.heads, shape.head_dim }) {
        if (result > std::numeric_limits<size_t>::max() / static_cast<size_t>(factor)) {
            throw std::overflow_error("post-RoPE Q/K tensor size overflow");
        }
        result *= static_cast<size_t>(factor);
    }
    return result;
}

inline size_t d2f_qk_canonical_index(
        const d2f_qk_tensor_shape & shape,
        int32_t token,
        int32_t head,
        int32_t dim) {
    if (token < 0 || token >= shape.tokens || head < 0 || head >= shape.heads ||
        dim < 0 || dim >= shape.head_dim) {
        throw std::out_of_range("post-RoPE Q/K host index is out of range");
    }
    return (static_cast<size_t>(token) * shape.heads + head) * shape.head_dim + dim;
}

inline std::vector<float> copy_d2f_qk_to_canonical_host(
        const float * source,
        const d2f_qk_tensor_shape & shape) {
    const size_t elements = d2f_qk_element_count(shape);
    if (!source) {
        throw std::invalid_argument("post-RoPE Q/K source is null");
    }
    return std::vector<float>(source, source + elements);
}

inline float d2f_qk_host_tensor::at(int32_t token, int32_t head, int32_t dim) const {
    return values.at(d2f_qk_canonical_index(shape, token, head, dim));
}

inline d2f_qk_tensor_shape validate_d2f_post_rope_qk_tensor(
        const ggml_tensor & tensor,
        d2f_qk_tensor_kind kind,
        int32_t expected_tokens,
        const d2f_qk_capture_config & config) {
    validate_d2f_qk_capture_config(config);
    if (!config.enabled || expected_tokens <= 0) {
        throw std::invalid_argument("post-RoPE Q/K capture is inactive or has no tokens");
    }
    if (tensor.type != GGML_TYPE_F32) {
        throw std::invalid_argument("post-RoPE Q/K tensor must be F32");
    }
    const int32_t expected_heads = kind == d2f_qk_tensor_kind::query
            ? config.query_heads
            : config.kv_heads;
    d2f_qk_tensor_shape shape;
    if (tensor.ne[0] == static_cast<int64_t>(config.head_dim) * expected_heads &&
        tensor.ne[1] == expected_tokens && tensor.ne[2] == 1 && tensor.ne[3] == 1) {
        // LLaDA keeps post-RoPE heads flattened as
        // [head_dim * heads][tokens]. Head slices remain contiguous.
        shape = { expected_tokens, expected_heads, config.head_dim };
    } else if (tensor.ne[0] == config.head_dim && tensor.ne[1] == expected_heads &&
               tensor.ne[2] == expected_tokens && tensor.ne[3] == 1) {
        shape = { expected_tokens, expected_heads, config.head_dim };
    } else {
        throw std::invalid_argument(
                "post-RoPE Q/K tensor shape [" + std::to_string(tensor.ne[0]) + "," +
                std::to_string(tensor.ne[1]) + "," + std::to_string(tensor.ne[2]) +
                "] does not match flattened or explicit-head capture layouts");
    }
    if (shape.tokens != expected_tokens || shape.heads != expected_heads ||
        shape.head_dim != config.head_dim) {
        throw std::invalid_argument(
                "post-RoPE Q/K tensor shape [" + std::to_string(shape.head_dim) + "," +
                std::to_string(shape.heads) + "," + std::to_string(shape.tokens) +
                "] does not match expected [" + std::to_string(config.head_dim) + "," +
                std::to_string(expected_heads) + "," + std::to_string(expected_tokens) + "]");
    }

    const size_t elements = d2f_qk_element_count(shape);
    if (!ggml_is_contiguous(&tensor) || tensor.nb[0] != sizeof(float) ||
        ggml_nbytes(&tensor) != elements * sizeof(float)) {
        throw std::invalid_argument("post-RoPE Q/K tensor must be densely contiguous");
    }
    return shape;
}

class d2f_post_rope_qk_capture {
  public:
    explicit d2f_post_rope_qk_capture(d2f_qk_capture_config config) : config_(config) {
        validate_d2f_qk_capture_config(config_);
    }

    void reset() {
        phase_ = d2f_qk_capture_phase::inactive;
        expected_tokens_ = 0;
        records_.clear();
        error_.clear();
    }

    void begin_phase(d2f_qk_capture_phase phase, int32_t expected_tokens) {
        if (!config_.enabled || phase == d2f_qk_capture_phase::inactive || expected_tokens <= 0) {
            throw std::invalid_argument("invalid post-RoPE Q/K capture phase");
        }
        if (phase_ != d2f_qk_capture_phase::inactive) {
            throw std::logic_error("post-RoPE Q/K capture phase is already active");
        }
        throw_if_failed();
        const auto existing = std::find_if(records_.begin(), records_.end(), [phase](const auto & record) {
            return record.phase == phase;
        });
        if (existing != records_.end()) {
            throw std::logic_error("post-RoPE Q/K capture phase was already recorded");
        }
        phase_ = phase;
        expected_tokens_ = expected_tokens;
    }

    void finish_phase() {
        if (phase_ == d2f_qk_capture_phase::inactive) {
            throw std::logic_error("post-RoPE Q/K capture phase is not active");
        }
        const d2f_qk_capture_phase finished_phase = phase_;
        phase_ = d2f_qk_capture_phase::inactive;
        expected_tokens_ = 0;
        throw_if_failed();
        for (int32_t layer = config_.total_layers - config_.last_n_layers;
             layer < config_.total_layers;
             ++layer) {
            const auto require_kind = [&](d2f_qk_tensor_kind kind) {
                if (!find(finished_phase, kind, layer)) {
                    throw std::runtime_error(
                            std::string("missing post-RoPE ") + d2f_qk_tensor_kind_name(kind) +
                            " tensor for " + d2f_qk_capture_phase_name(finished_phase) +
                            " phase layer " + std::to_string(layer));
                }
            };
            if (finished_phase == d2f_qk_capture_phase::prompt) {
                require_kind(d2f_qk_tensor_kind::query);
            }
            require_kind(d2f_qk_tensor_kind::key);
        }
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * user_data) {
        if (!user_data) {
            return false;
        }
        return static_cast<d2f_post_rope_qk_capture *>(user_data)->on_eval(tensor, ask);
    }

    const d2f_qk_host_tensor * find(
            d2f_qk_capture_phase phase,
            d2f_qk_tensor_kind kind,
            int32_t layer_index) const {
        const auto found = std::find_if(records_.begin(), records_.end(), [&](const auto & record) {
            return record.phase == phase && record.kind == kind && record.layer_index == layer_index;
        });
        return found == records_.end() ? nullptr : &*found;
    }

    const std::vector<d2f_qk_host_tensor> & records() const {
        return records_;
    }

    bool failed() const {
        return !error_.empty();
    }

    const std::string & error_message() const {
        return error_;
    }

    void throw_if_failed() const {
        if (failed()) {
            throw std::runtime_error(error_);
        }
    }

  private:
    bool on_eval(ggml_tensor * tensor, bool ask) {
        if (!config_.enabled || phase_ == d2f_qk_capture_phase::inactive || failed() || !tensor) {
            return false;
        }
        const auto parsed = parse_d2f_post_rope_qk_name(tensor->name);
        const bool post_rope = tensor->op == GGML_OP_ROPE;
        const bool phase_accepts_kind = post_rope && parsed &&
                (phase_ != d2f_qk_capture_phase::image || parsed->kind == d2f_qk_tensor_kind::key);
        const bool selected = phase_accepts_kind &&
                d2f_qk_layer_selected(config_, parsed->layer_index);
        if (ask || !selected) {
            return ask ? selected : true;
        }

        try {
            if (find(phase_, parsed->kind, parsed->layer_index)) {
                throw std::runtime_error("duplicate post-RoPE Q/K tensor");
            }
            const d2f_qk_tensor_shape shape = validate_d2f_post_rope_qk_tensor(
                    *tensor, parsed->kind, expected_tokens_, config_);
            const size_t elements = d2f_qk_element_count(shape);
            d2f_qk_host_tensor record = {
                phase_,
                parsed->kind,
                parsed->layer_index,
                shape,
                std::vector<float>(elements),
            };
            if (tensor->buffer) {
                ggml_backend_tensor_get(
                        tensor,
                        record.values.data(),
                        0,
                        elements * sizeof(float));
            } else if (tensor->data) {
                std::memcpy(
                        record.values.data(),
                        tensor->data,
                        elements * sizeof(float));
            } else {
                throw std::runtime_error("post-RoPE Q/K tensor has no readable storage");
            }
            records_.push_back(std::move(record));
            return true;
        } catch (const std::exception & exception) {
            error_ = std::string("post-RoPE Q/K capture failed for ") + tensor->name + ": " + exception.what();
            return false;
        }
    }

    d2f_qk_capture_config config_;
    d2f_qk_capture_phase phase_ = d2f_qk_capture_phase::inactive;
    int32_t expected_tokens_ = 0;
    std::vector<d2f_qk_host_tensor> records_;
    std::string error_;
};

} // namespace lladao::detail
