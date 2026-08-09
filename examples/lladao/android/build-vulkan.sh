#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
build_dir="${LLADAO_ANDROID_BUILD_DIR:-${repo_root}/build-android-vulkan}"
android_abi="${ANDROID_ABI:-arm64-v8a}"
android_platform="${ANDROID_PLATFORM:-android-28}"
android_ndk="${ANDROID_NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
cmake_prefix_path="${LLADAO_CMAKE_PREFIX_PATH:-${CMAKE_PREFIX_PATH:-}}"
spirv_headers_dir="${LLADAO_SPIRV_HEADERS_DIR:-}"
if [[ -z "${spirv_headers_dir}" && -n "${cmake_prefix_path}" &&
      -f "${cmake_prefix_path}/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]]; then
    spirv_headers_dir="${cmake_prefix_path}/share/cmake/SPIRV-Headers"
fi

if [[ -z "${android_ndk}" ]]; then
    echo "Set ANDROID_NDK, ANDROID_NDK_HOME, or ANDROID_NDK_ROOT." >&2
    exit 1
fi

toolchain="${android_ndk}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${toolchain}" ]]; then
    echo "Android NDK toolchain not found: ${toolchain}" >&2
    exit 1
fi

ndk_prebuilt="$(find "${android_ndk}/toolchains/llvm/prebuilt" -mindepth 1 -maxdepth 1 -type d -print -quit)"
vulkan_include="${VULKAN_INCLUDE_DIR:-${ndk_prebuilt}/sysroot/usr/include}"
vulkan_glslc="${VULKAN_GLSLC_EXECUTABLE:-$(find "${android_ndk}/shader-tools" -type f -name glslc -print -quit)}"
if [[ ! -f "${vulkan_include}/vulkan/vulkan.h" || ! -f "${vulkan_include}/vulkan/vulkan.hpp" ]]; then
    echo "Vulkan C and C++ headers not found under: ${vulkan_include}" >&2
    echo "Set VULKAN_INCLUDE_DIR to a Khronos Vulkan-Headers include directory." >&2
    exit 1
fi
if [[ ! -x "${vulkan_glslc}" ]]; then
    echo "glslc not found or not executable: ${vulkan_glslc}" >&2
    exit 1
fi

if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    jobs="$(sysctl -n hw.ncpu)"
else
    jobs=4
fi
jobs="${LLADAO_BUILD_JOBS:-${jobs}}"

cmake_args=(
    -S "${repo_root}"
    -B "${build_dir}"
    -DCMAKE_TOOLCHAIN_FILE="${toolchain}" \
    -DANDROID_ABI="${android_abi}" \
    -DANDROID_PLATFORM="${android_platform}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_VULKAN=ON \
    -DVulkan_GLSLC_EXECUTABLE="${vulkan_glslc}" \
    -DVulkan_INCLUDE_DIR="${vulkan_include}" \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_MTMD=ON \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_OPENSSL=OFF
)
if [[ -n "${cmake_prefix_path}" ]]; then
    cmake_args+=("-DCMAKE_PREFIX_PATH=${cmake_prefix_path}")
fi
if [[ -n "${spirv_headers_dir}" ]]; then
    cmake_args+=("-DSPIRV-Headers_DIR=${spirv_headers_dir}")
fi

cmake "${cmake_args[@]}"

cmake --build "${build_dir}" \
    --target \
        llama-lladao-d2f \
        test-backend-ops \
        test-diffusion-d2f-scheduler \
        test-mtmd-lladao \
    -j "${jobs}"

echo "Android Vulkan binaries are in ${build_dir}/bin"
