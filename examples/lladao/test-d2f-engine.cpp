#include "lladao-d2f-engine.h"
#include "lladao-d2f-decode.h"
#include "lladao-d2f-prefill.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

static void require_invalid(const lladao::d2f_engine_params & params) {
    try {
        lladao::d2f_engine engine(params);
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid engine parameters were accepted");
}

static void require_configuration_valid(const lladao::d2f_engine_params & params) {
    try {
        lladao::d2f_engine engine(params);
    } catch (const std::invalid_argument &) {
        throw std::runtime_error("valid engine parameters were rejected");
    } catch (...) {
        return;
    }
}

static void require_invalid_cpu_mask(const char * value) {
    try {
        (void) lladao::detail::parse_cpu_mask_hex(value);
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid CPU mask was accepted");
}

int main() {
    static_assert(!std::is_copy_constructible<lladao::d2f_engine>::value, "engine must not be copied");
    static_assert(std::is_nothrow_move_constructible<lladao::d2f_engine>::value, "engine must be movable");

    lladao::d2f_engine_params params;
    if (params.prefix_prefill_mode != lladao::d2f_prefix_prefill_mode::exact ||
        params.prefix_pack_size != 512 || params.release_vision_after_encode) {
        throw std::runtime_error("default prefix prefill settings changed");
    }
    if (params.flash_attention_mode != lladao::d2f_flash_attention_mode::auto_select) {
        throw std::runtime_error("default Flash Attention mode must remain auto");
    }
    if (params.cache_type_k != GGML_TYPE_F16 || params.cache_type_v != GGML_TYPE_F16) {
        throw std::runtime_error("default D2F cache types must remain f16");
    }
    if (params.cpu_mask != 0 || params.cpu_strict || params.cpu_poll != 50) {
        throw std::runtime_error("default D2F CPU affinity settings changed");
    }
    if (params.generation_block_cache || params.sparse_generation_logits) {
        throw std::runtime_error("experimental D2F generation optimizations must default off");
    }
    if (std::string(lladao::detail::language_device_mode(0)) != "cpu_only" ||
        std::string(lladao::detail::language_device_mode(1)) != "automatic" ||
        std::string(lladao::detail::language_device_mode(-1)) != "automatic") {
        throw std::runtime_error("D2F language device mode resolution is invalid");
    }
    if (lladao::detail::parse_cpu_mask_hex("0xfe") != UINT64_C(0xfe) ||
        lladao::detail::parse_cpu_mask_hex("FE") != UINT64_C(0xfe) ||
        lladao::detail::format_cpu_mask(UINT64_C(0xfe)) != "0xfe" ||
        lladao::detail::resolve_cpu_threads(UINT64_C(0xfe), 0) != 7 ||
        lladao::detail::resolve_cpu_threads(UINT64_C(0xfe), 4) != 4 ||
        lladao::detail::resolve_cpu_threads(0, 0) != 0) {
        throw std::runtime_error("D2F CPU affinity parsing or resolution is invalid");
    }
    require_invalid_cpu_mask("");
    require_invalid_cpu_mask("0x");
    require_invalid_cpu_mask("0xfg");
    require_invalid_cpu_mask("0x10000000000000000");

    {
        ggml_backend_load_all();
        ggml_backend_dev_t cpu_device =
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_reg_t cpu_registry =
                cpu_device ? ggml_backend_dev_backend_reg(cpu_device) : nullptr;
        auto * new_fn = cpu_registry
                ? reinterpret_cast<decltype(ggml_threadpool_new) *>(
                        ggml_backend_reg_get_proc_address(cpu_registry, "ggml_threadpool_new"))
                : nullptr;
        auto * free_fn = cpu_registry
                ? reinterpret_cast<decltype(ggml_threadpool_free) *>(
                        ggml_backend_reg_get_proc_address(cpu_registry, "ggml_threadpool_free"))
                : nullptr;
        auto * resume_fn = cpu_registry
                ? reinterpret_cast<decltype(ggml_threadpool_resume) *>(
                        ggml_backend_reg_get_proc_address(cpu_registry, "ggml_threadpool_resume"))
                : nullptr;
        auto * pause_fn = cpu_registry
                ? reinterpret_cast<decltype(ggml_threadpool_pause) *>(
                        ggml_backend_reg_get_proc_address(cpu_registry, "ggml_threadpool_pause"))
                : nullptr;
        if (!new_fn || !free_fn || !resume_fn || !pause_fn) {
            throw std::runtime_error("CPU backend threadpool API is unavailable");
        }
        ggml_threadpool_params threadpool_params = ggml_threadpool_params_default(2);
        threadpool_params.paused = true;
        std::unique_ptr<ggml_threadpool, decltype(free_fn)> threadpool(
                new_fn(&threadpool_params),
                free_fn);
        if (!threadpool) {
            throw std::runtime_error("ggml threadpool lifecycle smoke failed");
        }
        resume_fn(threadpool.get());
        pause_fn(threadpool.get());
    }

    {
        const int32_t mask = 99;
        const std::vector<int32_t> first_tokens = { 1, 2, mask, mask, mask, mask, mask, mask };
        const auto first = lladao::detail::plan_d2f_generation_decode(
                first_tokens, mask, 0, 4, 4, 0, true, true);
        if (first.first_position != 0 || first.input_rows() != 4 ||
            first.rebuild_rows() != 0 || first.logit_positions != std::vector<int32_t>({ 2, 3 }) ||
            first.batch_index(3) != 3 || !first.requests_logits(2) || first.requests_logits(1)) {
            throw std::runtime_error("initial sparse D2F decode plan is invalid");
        }

        const std::vector<int32_t> transition_tokens = { 1, 2, 3, 4, mask, mask, mask, mask };
        const auto transition = lladao::detail::plan_d2f_generation_decode(
                transition_tokens, mask, 4, 8, 4, 0, true, true);
        if (transition.first_position != 0 || transition.next_cached_complete_end != 4 ||
            transition.input_rows() != 8 || transition.active_rows() != 4 ||
            transition.rebuild_rows() != 4 ||
            transition.logit_positions != std::vector<int32_t>({ 4, 5, 6, 7 })) {
            throw std::runtime_error("finalized-block D2F rebuild plan is invalid");
        }

        const auto cached = lladao::detail::plan_d2f_generation_decode(
                transition_tokens, mask, 4, 8, 4, 4, true, true);
        if (cached.first_position != 4 || cached.input_rows() != 4 ||
            cached.rebuild_rows() != 0 || cached.batch_index(7) != 3) {
            throw std::runtime_error("cached D2F active suffix plan is invalid");
        }

        const auto dense = lladao::detail::plan_d2f_generation_decode(
                transition_tokens, mask, 4, 8, 4, 4, true, false);
        if (dense.logit_positions != std::vector<int32_t>({ 4, 5, 6, 7 })) {
            throw std::runtime_error("dense cached D2F logits plan is invalid");
        }

        const auto full = lladao::detail::plan_d2f_generation_decode(
                transition_tokens, mask, 4, 8, 4, 4, false, false);
        if (full.first_position != 0 || full.next_cached_complete_end != 0 ||
            full.rebuild_rows() != 4 || full.logit_positions.size() != 8) {
            throw std::runtime_error("full-recompute dense D2F plan is invalid");
        }

        lladao::detail::d2f_decode_counters counters;
        counters.add(first);
        counters.add(transition);
        counters.add(cached);
        if (counters.iterations != 3 || counters.input_rows != 16 ||
            counters.active_rows != 12 || counters.rebuild_rows != 4 ||
            counters.logit_rows != 10 || counters.reused_input_rows != 4) {
            throw std::runtime_error("D2F decode counters are invalid");
        }

        bool rejected = false;
        try {
            (void) lladao::detail::plan_d2f_generation_decode(
                    first_tokens, mask, 4, 8, 4, 0, true, true);
        } catch (const std::logic_error &) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("masked finalized D2F block was accepted");
        }
    }

    {
        const auto chunks = lladao::detail::plan_prefix_chunks(
                { 1858 },
                403,
                lladao::d2f_prefix_prefill_mode::exact,
                512);
        if (chunks.size() != 1 || chunks[0].offset != 0 || chunks[0].length != 2261 ||
            chunks[0].kind != lladao::detail::prefix_chunk_kind::complete ||
            lladao::detail::prefix_chunk_ubatch(chunks, 64) != 2261) {
            throw std::runtime_error("exact prefix chunk plan is invalid");
        }
    }

    {
        const auto chunks = lladao::detail::plan_prefix_chunks(
                { 1858, 100 },
                403,
                lladao::d2f_prefix_prefill_mode::component_exact,
                512);
        if (chunks.size() != 3 ||
            chunks[0].offset != 0 || chunks[0].length != 1858 ||
            chunks[1].offset != 1858 || chunks[1].length != 100 ||
            chunks[2].offset != 1958 || chunks[2].length != 403 ||
            chunks[2].kind != lladao::detail::prefix_chunk_kind::prompt ||
            lladao::detail::prefix_chunk_ubatch(chunks, 64) != 1858) {
            throw std::runtime_error("component exact prefix chunk plan is invalid");
        }
    }

    {
        const auto chunks = lladao::detail::plan_prefix_chunks(
                { 1858, 100 },
                403,
                lladao::d2f_prefix_prefill_mode::component_parallel,
                512);
        if (chunks.size() != 2 ||
            chunks[0].offset != 0 || chunks[0].length != 1958 ||
            chunks[0].kind != lladao::detail::prefix_chunk_kind::image ||
            chunks[0].component != -1 ||
            chunks[1].offset != 1958 || chunks[1].length != 403 ||
            chunks[1].kind != lladao::detail::prefix_chunk_kind::prompt ||
            lladao::detail::prefix_chunk_ubatch(chunks, 64) != 1958) {
            throw std::runtime_error("parallel component prefix chunk plan is invalid");
        }
    }

    {
        const auto chunks = lladao::detail::plan_prefix_chunks(
                { 1858 },
                403,
                lladao::d2f_prefix_prefill_mode::packed_image,
                512);
        const std::vector<int32_t> expected_lengths = { 512, 512, 512, 322, 403 };
        if (chunks.size() != expected_lengths.size() ||
            lladao::detail::prefix_chunk_ubatch(chunks, 64) != 512) {
            throw std::runtime_error("packed image prefix chunk capacity is invalid");
        }
        int32_t offset = 0;
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (chunks[i].offset != offset || chunks[i].length != expected_lengths[i]) {
                throw std::runtime_error("packed image prefix chunk boundary is invalid");
            }
            offset += chunks[i].length;
        }
    }

    {
        std::unique_ptr<int> vision = std::make_unique<int>(1);
        if (lladao::detail::release_vision_context(vision, false) || !vision) {
            throw std::runtime_error("disabled vision release changed the context");
        }
        if (!lladao::detail::release_vision_context(vision, true) || vision ||
            !lladao::detail::vision_context_needs_rebuild(false, false, false)) {
            throw std::runtime_error("enabled vision release did not require a rebuild");
        }
        vision = std::make_unique<int>(2);
        if (lladao::detail::vision_context_needs_rebuild(true, true, true) ||
            !lladao::detail::vision_context_needs_rebuild(true, true, false)) {
            throw std::runtime_error("vision context mode rebuild decision is invalid");
        }
    }

    require_invalid(params);

    params.model_path = "unused.gguf";
    params.mmproj_path = "unused-mmproj.gguf";
    params.context_size = 0;
    require_invalid(params);

    params.context_size = 16384;
    params.max_iterations = 0;
    require_invalid(params);

    params.max_iterations = 256;
    params.cpu_poll = 101;
    require_invalid(params);

    params.cpu_poll = 50;
    params.cpu_mask = UINT64_C(1);
    params.threads = GGML_MAX_N_THREADS + 1;
    require_invalid(params);

    params.cpu_mask = 0;
    params.threads = 0;
    params.prefix_pack_size = 0;
    require_invalid(params);

    params.prefix_pack_size = 512;
    params.prefix_prefill_mode = static_cast<lladao::d2f_prefix_prefill_mode>(99);
    require_invalid(params);

    params.prefix_prefill_mode = lladao::d2f_prefix_prefill_mode::component_exact;
    params.prefix_cache = false;
    require_invalid(params);

    params.prefix_cache = true;
    params.prefix_prefill_mode = lladao::d2f_prefix_prefill_mode::exact;
    params.flash_attention_mode = static_cast<lladao::d2f_flash_attention_mode>(99);
    require_invalid(params);

    params.flash_attention_mode = lladao::d2f_flash_attention_mode::auto_select;
    params.yarn_factor = 0.5f;
    require_invalid(params);

    params.yarn_factor = 1.0f;
    params.cache_type_k = GGML_TYPE_F32;
    require_invalid(params);

    params.cache_type_k = GGML_TYPE_F16;
    params.cache_type_v = GGML_TYPE_Q4_0;
    require_invalid(params);

    params.cache_type_v = GGML_TYPE_Q8_0;
    params.flash_attention_mode = lladao::d2f_flash_attention_mode::auto_select;
    require_invalid(params);

    params.flash_attention_mode = lladao::d2f_flash_attention_mode::disabled;
    require_invalid(params);

    params.flash_attention_mode = lladao::d2f_flash_attention_mode::enabled;
    require_configuration_valid(params);

    params.cache_type_v = GGML_TYPE_F16;
    params.flash_attention_mode = lladao::d2f_flash_attention_mode::disabled;
    require_configuration_valid(params);

    params.prefix_prefill_mode = lladao::d2f_prefix_prefill_mode::component_parallel;
    require_invalid(params);

    params.flash_attention_mode = lladao::d2f_flash_attention_mode::auto_select;
    require_configuration_valid(params);

    return 0;
}
