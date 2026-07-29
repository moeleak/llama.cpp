set(TARGET llama-lladao-d2f)
add_executable(${TARGET} lladao-d2f.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ../diffusion)
target_link_libraries(${TARGET} PRIVATE
    llama-diffusion
    llama
    llama-common
    mtmd
    ggml
    ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
