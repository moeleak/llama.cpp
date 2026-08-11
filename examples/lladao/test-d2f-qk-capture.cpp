#include "lladao-d2f-qk-capture.h"

#include <cstring>
#include <stdexcept>
#include <vector>

using namespace lladao::detail;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Function>
static void require_throws(Function && function, const char * message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, message);
}

static ggml_tensor make_tensor(
        const char * name,
        ggml_type type,
        int32_t head_dim,
        int32_t heads,
        int32_t tokens,
        void * data) {
    ggml_tensor tensor = {};
    tensor.type = type;
    tensor.op = GGML_OP_ROPE;
    tensor.ne[0] = head_dim;
    tensor.ne[1] = heads;
    tensor.ne[2] = tokens;
    tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = static_cast<size_t>(head_dim) * tensor.nb[0];
    tensor.nb[2] = static_cast<size_t>(heads) * tensor.nb[1];
    tensor.nb[3] = static_cast<size_t>(tokens) * tensor.nb[2];
    tensor.data = data;
    std::strncpy(tensor.name, name, GGML_MAX_NAME - 1);
    return tensor;
}

static ggml_tensor make_flattened_tensor(
        const char * name,
        int32_t head_dim,
        int32_t heads,
        int32_t tokens,
        void * data) {
    ggml_tensor tensor = {};
    tensor.type = GGML_TYPE_F32;
    tensor.op = GGML_OP_ROPE;
    tensor.ne[0] = static_cast<int64_t>(head_dim) * heads;
    tensor.ne[1] = tokens;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = static_cast<size_t>(tensor.ne[0]) * tensor.nb[0];
    tensor.nb[2] = static_cast<size_t>(tokens) * tensor.nb[1];
    tensor.nb[3] = tensor.nb[2];
    tensor.data = data;
    std::strncpy(tensor.name, name, GGML_MAX_NAME - 1);
    return tensor;
}

int main() {
    const auto query_name = parse_d2f_post_rope_qk_name("Qcur-0");
    const auto key_name = parse_d2f_post_rope_qk_name("Kcur-31");
    require(
            query_name && query_name->kind == d2f_qk_tensor_kind::query && query_name->layer_index == 0 &&
            key_name && key_name->kind == d2f_qk_tensor_kind::key && key_name->layer_index == 31,
            "post-RoPE Q/K tensor names were not parsed");
    for (const char * rejected : {
             "Qcur_normed-31", "Kcur_normed-31", "mtp_Qcur-31", "Qcur-01",
             "Qcur--1", "Qcur-31-extra", "qcur-31", "Qcur-2147483648",
         }) {
        require(!parse_d2f_post_rope_qk_name(rejected), "non-exact Q/K tensor name was accepted");
    }

    const d2f_qk_capture_config config = {
        true,
        32,
        2,
        2,
        1,
        3,
    };
    require(
            !d2f_qk_layer_selected(config, 29) && d2f_qk_layer_selected(config, 30) &&
            d2f_qk_layer_selected(config, 31) && !d2f_qk_layer_selected(config, 32),
            "last-N Q/K layer filter is invalid");

    const d2f_qk_tensor_shape shape = { 2, 2, 3 };
    const std::vector<float> native = {
        0.0f, 1.0f, 2.0f,
        10.0f, 11.0f, 12.0f,
        100.0f, 101.0f, 102.0f,
        110.0f, 111.0f, 112.0f,
    };
    d2f_qk_host_tensor host;
    host.shape = shape;
    host.values = copy_d2f_qk_to_canonical_host(native.data(), shape);
    require(
            host.at(0, 1, 2) == 12.0f && host.at(1, 0, 1) == 101.0f &&
            host.at(1, 1, 2) == 112.0f,
            "Q/K host tensor is not laid out as [token][head][dim]");

    std::vector<float> query_values = native;
    std::vector<float> key_values = {
        0.0f, 1.0f, 2.0f,
        100.0f, 101.0f, 102.0f,
    };
    ggml_tensor query = make_tensor("Qcur-31", GGML_TYPE_F32, 3, 2, 2, query_values.data());
    ggml_tensor key = make_tensor("Kcur-31", GGML_TYPE_F32, 3, 1, 2, key_values.data());
    ggml_tensor pre_rope_query = query;
    pre_rope_query.op = GGML_OP_MUL_MAT;
    require(
            !d2f_post_rope_qk_capture::callback(&pre_rope_query, true, nullptr),
            "null capture unexpectedly requested a pre-RoPE tensor");
    require(
            validate_d2f_post_rope_qk_tensor(
                    query, d2f_qk_tensor_kind::query, 2, config).heads == 2 &&
            validate_d2f_post_rope_qk_tensor(
                    key, d2f_qk_tensor_kind::key, 2, config).heads == 1,
            "valid post-RoPE Q/K tensor shape was rejected");
    ggml_tensor flattened_query = make_flattened_tensor(
            "Qcur-31", 3, 2, 2, query_values.data());
    require(
            validate_d2f_post_rope_qk_tensor(
                    flattened_query, d2f_qk_tensor_kind::query, 2, config).heads == 2,
            "valid flattened post-RoPE Q tensor shape was rejected");

    ggml_tensor strided = query;
    strided.nb[2] += sizeof(float);
    require_throws(
            [&] {
                validate_d2f_post_rope_qk_tensor(
                        strided, d2f_qk_tensor_kind::query, 2, config);
            },
            "strided post-RoPE Q/K tensor was accepted");
    ggml_tensor wrong_type = query;
    wrong_type.type = GGML_TYPE_F16;
    require_throws(
            [&] {
                validate_d2f_post_rope_qk_tensor(
                        wrong_type, d2f_qk_tensor_kind::query, 2, config);
            },
            "non-F32 post-RoPE Q/K tensor was accepted");

    d2f_post_rope_qk_capture capture(config);
    capture.begin_phase(d2f_qk_capture_phase::image, 2);
    ggml_set_name(&pre_rope_query, "Kcur-30");
    require(
            !d2f_post_rope_qk_capture::callback(&pre_rope_query, true, &capture),
            "pre-RoPE tensor was requested by the capture");
    ggml_set_name(&query, "Qcur-29");
    require(
            !d2f_post_rope_qk_capture::callback(&query, true, &capture),
            "unselected Q/K layer requested evaluation");
    ggml_set_name(&query, "Qcur-30");
    ggml_set_name(&key, "Kcur-30");
    require(
            !d2f_post_rope_qk_capture::callback(&query, true, &capture) &&
            d2f_post_rope_qk_capture::callback(&key, true, &capture) &&
            d2f_post_rope_qk_capture::callback(&key, false, &capture),
            "image phase did not filter Q or capture K");
    ggml_set_name(&query, "Qcur-31");
    ggml_set_name(&key, "Kcur-31");
    require(
            !d2f_post_rope_qk_capture::callback(&query, true, &capture) &&
            d2f_post_rope_qk_capture::callback(&key, true, &capture) &&
            d2f_post_rope_qk_capture::callback(&key, false, &capture),
            "last selected image K layer was not captured");
    capture.finish_phase();
    require(
            capture.records().size() == 2 &&
            capture.find(d2f_qk_capture_phase::image, d2f_qk_tensor_kind::query, 30) == nullptr &&
            capture.find(d2f_qk_capture_phase::image, d2f_qk_tensor_kind::key, 31)->at(1, 0, 1) == 101.0f,
            "captured image K host values are invalid");

    std::vector<float> prompt_query_values(query_values.begin(), query_values.begin() + 6);
    std::vector<float> prompt_key_values(key_values.begin(), key_values.begin() + 3);
    ggml_tensor prompt_query = make_tensor(
            "Qcur-30", GGML_TYPE_F32, 3, 2, 1, prompt_query_values.data());
    ggml_tensor prompt_key = make_tensor(
            "Kcur-30", GGML_TYPE_F32, 3, 1, 1, prompt_key_values.data());
    capture.begin_phase(d2f_qk_capture_phase::prompt, 1);
    for (int32_t layer : { 30, 31 }) {
        ggml_format_name(&prompt_query, "Qcur-%d", layer);
        ggml_format_name(&prompt_key, "Kcur-%d", layer);
        require(
                d2f_post_rope_qk_capture::callback(&prompt_query, true, &capture) &&
                d2f_post_rope_qk_capture::callback(&prompt_query, false, &capture) &&
                d2f_post_rope_qk_capture::callback(&prompt_key, true, &capture) &&
                d2f_post_rope_qk_capture::callback(&prompt_key, false, &capture),
                "prompt Q/K tensors were not captured");
    }
    capture.finish_phase();
    require(
            capture.records().size() == 6 &&
            capture.find(d2f_qk_capture_phase::prompt, d2f_qk_tensor_kind::query, 31)->shape.tokens == 1 &&
            capture.find(d2f_qk_capture_phase::prompt, d2f_qk_tensor_kind::key, 30)->at(0, 0, 2) == 2.0f,
            "prompt-phase Q/K capture is invalid");

    return 0;
}
