#include "lladao-d2f-engine.h"

#include <stdexcept>
#include <type_traits>

static void require_invalid(const lladao::d2f_engine_params & params) {
    try {
        lladao::d2f_engine engine(params);
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid engine parameters were accepted");
}

int main() {
    static_assert(!std::is_copy_constructible<lladao::d2f_engine>::value, "engine must not be copied");
    static_assert(std::is_nothrow_move_constructible<lladao::d2f_engine>::value, "engine must be movable");

    lladao::d2f_engine_params params;
    require_invalid(params);

    params.model_path = "unused.gguf";
    params.mmproj_path = "unused-mmproj.gguf";
    params.context_size = 0;
    require_invalid(params);

    params.context_size = 16384;
    params.max_iterations = 0;
    require_invalid(params);

    params.max_iterations = 256;
    params.yarn_factor = 0.5f;
    require_invalid(params);

    return 0;
}
