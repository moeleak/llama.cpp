#include "fast-dvlm-speculative.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#define CHECK(expr)                                           \
    do {                                                      \
        if (!(expr))                                          \
            throw std::runtime_error("check failed: " #expr); \
    } while (0)

int main() {
    constexpr llama_token mask = 99;

    {
        std::vector<llama_token>       tokens     = { 7, mask, mask, mask };
        const std::vector<llama_token> draft_rows = { 8, 9, 10, 11 };
        fast_dvlm::apply_draft_predictions(tokens, mask, 1, [&](int32_t row) { return draft_rows.at(row); });
        CHECK((tokens == std::vector<llama_token>{ 7, 8, 9, 10 }));

        const auto step = fast_dvlm::resolve_speculative_step(tokens, { 8, 9, 10, 11 }, 1);
        CHECK(step.accepted == 4);
        CHECK(step.output_count == 3);
        CHECK((step.output == std::vector<llama_token>{ 8, 9, 10 }));
        CHECK(step.inherited == 11);
    }

    {
        // Later blocks enter as all masks. The inherited token is inserted
        // after block_start is computed, so it is emitted exactly once.
        std::vector<llama_token> tokens(4, mask);
        const int32_t            block_start = 0;
        tokens[0]                            = 11;
        fast_dvlm::apply_draft_predictions(tokens, mask, 1,
                                           [](int32_t row) { return static_cast<llama_token>(12 + row); });
        CHECK((tokens == std::vector<llama_token>{ 11, 12, 13, 14 }));
        const auto step = fast_dvlm::resolve_speculative_step(tokens, { 12, 13, 77, 15 }, block_start);
        CHECK(step.accepted == 3);
        CHECK(step.output_count == 3);
        CHECK((step.output == std::vector<llama_token>{ 11, 12, 13 }));
        CHECK(step.inherited == 77);
    }

    {
        bool threw = false;
        try {
            (void) fast_dvlm::resolve_speculative_step({ 1 }, {}, 0);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    std::cout << "Fast-dVLM speculative scheduler tests passed\n";
    return 0;
}
