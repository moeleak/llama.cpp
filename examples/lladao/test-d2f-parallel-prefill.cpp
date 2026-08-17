#include "lladao-d2f-prefill.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

static std::vector<int64_t> span_local_weighted_sum(
        const lladao::detail::prefix_parallel_audit & audit,
        const std::vector<int64_t> & values) {
    std::vector<int64_t> result(values.size(), 0);
    for (int32_t query = 0; query < static_cast<int32_t>(values.size()); ++query) {
        for (int32_t key = 0; key < static_cast<int32_t>(values.size()); ++key) {
            if (lladao::detail::prefix_attention_allowed(audit, query, key)) {
                result[query] += static_cast<int64_t>(key + 1) * values[key];
            }
        }
    }
    return result;
}

int main() {
    {
        bool packed_enabled = false;
        int32_t clear_calls = 0;
        const int32_t value = lladao::detail::with_packed_prefill_contract(
                [&]() {
                    if (packed_enabled) {
                        throw std::runtime_error("packed contract enabled twice");
                    }
                    packed_enabled = true;
                },
                [&]() noexcept {
                    packed_enabled = false;
                    ++clear_calls;
                },
                []() { return 7; });
        if (value != 7 || packed_enabled || clear_calls != 1) {
            throw std::runtime_error("packed contract did not clear after success");
        }

        bool caught = false;
        try {
            lladao::detail::with_packed_prefill_contract(
                    [&]() { packed_enabled = true; },
                    [&]() noexcept {
                        packed_enabled = false;
                        ++clear_calls;
                    },
                    []() -> int32_t {
                        throw std::runtime_error("injected packed decode failure");
                    });
        } catch (const std::runtime_error &) {
            caught = true;
        }
        if (!caught || packed_enabled || clear_calls != 2) {
            throw std::runtime_error("packed contract did not clear after an exception");
        }
    }

    const std::vector<int32_t> image_lengths = { 3, 5, 2 };
    const auto audit = lladao::detail::audit_prefix_parallelism(
            image_lengths,
            lladao::d2f_prefix_prefill_mode::component_parallel,
            true);
    if (audit.cu_seqlens != std::vector<int32_t>({ 0, 3, 8, 10 }) ||
        audit.domains != 3 ||
        audit.image_decode_calls != 1 ||
        audit.attention_pairs_dense != 100 ||
        audit.attention_pairs_packed != 38 ||
        !audit.flash_attention_resolved ||
        !audit.batched_submission) {
        throw std::runtime_error("three-span parallel prefix metadata is invalid");
    }

    const auto chunks = lladao::detail::plan_prefix_chunks(
            image_lengths,
            7,
            lladao::d2f_prefix_prefill_mode::component_parallel,
            512);
    if (chunks.size() != 2 ||
        chunks[0].offset != 0 || chunks[0].length != 10 ||
        chunks[0].kind != lladao::detail::prefix_chunk_kind::image ||
        chunks[1].offset != 10 || chunks[1].length != 7 ||
        chunks[1].kind != lladao::detail::prefix_chunk_kind::prompt) {
        throw std::runtime_error("three-span component_parallel chunk plan is invalid");
    }

    int64_t allowed_pairs = 0;
    for (int32_t query = 0; query < 10; ++query) {
        for (int32_t key = 0; key < 10; ++key) {
            const bool expected =
                    (query < 3 && key < 3) ||
                    (query >= 3 && query < 8 && key >= 3 && key < 8) ||
                    (query >= 8 && key >= 8);
            const bool allowed = lladao::detail::prefix_attention_allowed(audit, query, key);
            if (allowed != expected) {
                throw std::runtime_error("packed attention crossed an image-span boundary");
            }
            allowed_pairs += allowed;
        }
    }
    if (allowed_pairs != audit.attention_pairs_packed) {
        throw std::runtime_error("packed attention pair count is invalid");
    }

    const std::vector<int64_t> original = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<int64_t> mutated = original;
    for (int32_t index = 3; index < 8; ++index) {
        mutated[index] += 1000;
    }
    const std::vector<int64_t> original_output = span_local_weighted_sum(audit, original);
    const std::vector<int64_t> mutated_output = span_local_weighted_sum(audit, mutated);
    for (int32_t index = 0; index < 10; ++index) {
        const bool belongs_to_mutated_span = index >= 3 && index < 8;
        if ((original_output[index] != mutated_output[index]) != belongs_to_mutated_span) {
            throw std::runtime_error("span-local mutation isolation is invalid");
        }
    }

    const auto unresolved = lladao::detail::audit_prefix_parallelism(
            image_lengths,
            lladao::d2f_prefix_prefill_mode::component_parallel,
            false);
    if (!unresolved.batched_submission || unresolved.image_decode_calls != 1 ||
        unresolved.cu_seqlens != audit.cu_seqlens ||
        unresolved.attention_pairs_packed != audit.attention_pairs_packed) {
        throw std::runtime_error("unresolved Flash Attention audit is invalid");
    }
    bool rejected = false;
    try {
        lladao::detail::validate_prefix_parallel_backend(
                lladao::d2f_prefix_prefill_mode::component_parallel,
                false);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("component_parallel accepted unresolved Flash Attention");
    }
    lladao::detail::validate_prefix_parallel_backend(
            lladao::d2f_prefix_prefill_mode::component_parallel,
            true);
    lladao::detail::validate_prefix_parallel_backend(
            lladao::d2f_prefix_prefill_mode::component_exact,
            false);

    const auto packed_parallel = lladao::detail::audit_prefix_parallelism(
            { 10 },
            lladao::d2f_prefix_prefill_mode::packed_parallel,
            true,
            4);
    if (packed_parallel.cu_seqlens != std::vector<int32_t>({ 0, 4, 8, 10 }) ||
        packed_parallel.domains != 3 || packed_parallel.image_decode_calls != 1 ||
        packed_parallel.attention_pairs_dense != 100 ||
        packed_parallel.attention_pairs_packed != 36 ||
        !packed_parallel.batched_submission ||
        lladao::detail::prefix_attention_allowed(packed_parallel, 3, 4) ||
        !lladao::detail::prefix_attention_allowed(packed_parallel, 4, 7)) {
        throw std::runtime_error("single-image packed_parallel visibility is invalid");
    }
    rejected = false;
    try {
        lladao::detail::validate_prefix_parallel_backend(
                lladao::d2f_prefix_prefill_mode::packed_parallel,
                false);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("packed_parallel accepted unresolved Flash Attention");
    }
    lladao::detail::validate_prefix_parallel_backend(
            lladao::d2f_prefix_prefill_mode::packed_parallel,
            true);

    // Packed image shape is request-local. Different token counts, maximum
    // lane lengths, and lane counts reuse one language context whenever its
    // resident, batch, output, and sequence capacities cover both requests.
    const lladao::detail::language_context_capacity reusable_capacity = {
        128,
        64,
        16,
        4,
    };
    const auto second_audit = lladao::detail::audit_prefix_parallelism(
            { 4, 2, 6, 1 },
            lladao::d2f_prefix_prefill_mode::component_parallel,
            true);
    if (lladao::detail::language_context_needs_rebuild(
                true,
                reusable_capacity,
                96,
                audit.cu_seqlens.back(),
                8,
                audit.domains) ||
        lladao::detail::language_context_needs_rebuild(
                true,
                reusable_capacity,
                96,
                second_audit.cu_seqlens.back(),
                8,
                second_audit.domains)) {
        throw std::runtime_error("packed request shape incorrectly became context identity");
    }
    if (!lladao::detail::language_context_needs_rebuild(
                true,
                reusable_capacity,
                96,
                65,
                8,
                second_audit.domains) ||
        !lladao::detail::language_context_needs_rebuild(
                false,
                reusable_capacity,
                96,
                audit.cu_seqlens.back(),
                8,
                audit.domains)) {
        throw std::runtime_error("language context capacity rebuild decision is invalid");
    }

    return 0;
}
