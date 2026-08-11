set(ENGINE_TARGET llama-lladao-d2f-engine)
add_library(${ENGINE_TARGET} STATIC lladao-d2f.cpp lladao-d2f-decode.h lladao-d2f-engine.h)
target_include_directories(${ENGINE_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${ENGINE_TARGET} PRIVATE ../diffusion)
target_link_libraries(${ENGINE_TARGET} PUBLIC
    llama-diffusion
    llama
    llama-common
    mtmd
    ggml
    ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${ENGINE_TARGET} PUBLIC cxx_std_17)
set_target_properties(${ENGINE_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TARGET llama-lladao-d2f)
add_executable(${TARGET} lladao-d2f-main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ${ENGINE_TARGET})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS)
    set(TEST_TARGET test-lladao-d2f-engine)
    add_executable(${TEST_TARGET} test-d2f-engine.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ${ENGINE_TARGET})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})

    set(TEST_TARGET test-lladao-d2f-kv-compression)
    add_executable(${TEST_TARGET} test-d2f-kv-compression.cpp)
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})

    set(TEST_TARGET test-lladao-d2f-qk-capture)
    add_executable(${TEST_TARGET} test-d2f-qk-capture.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})

    set(TEST_TARGET test-lladao-d2f-parallel-prefill)
    add_executable(${TEST_TARGET} test-d2f-parallel-prefill.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
endif()
