set(TARGET llama-fast-dvlm)
add_executable(${TARGET} fast-dvlm.cpp fast-dvlm-speculative.h)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE
    llama
    llama-common
    mtmd
    ggml
    ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS)
    set(TEST_TARGET test-fast-dvlm-speculative)
    add_executable(${TEST_TARGET} test-fast-dvlm-speculative.cpp fast-dvlm-speculative.h)
    target_link_libraries(${TEST_TARGET} PRIVATE llama)
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
endif()
