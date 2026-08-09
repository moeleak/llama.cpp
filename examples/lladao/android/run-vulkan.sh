#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
build_dir="${LLADAO_ANDROID_BUILD_DIR:-${repo_root}/build-android-vulkan}"
remote_dir="${LLADAO_ANDROID_REMOTE_DIR:-/data/local/tmp/lladao-vulkan}"
adb_bin="${ADB:-adb}"

model=""
mmproj=""
lora=""
image=""
prompt=""
ctx_size=8192
gpu_layers=0
threads=4
max_iterations=256
vision_gpu=1
preprocess_only=0
prefix_cache=1

usage() {
    cat >&2 <<EOF
Usage: $0 --model MODEL.gguf --mmproj MMPROJ.gguf --image IMAGE --prompt TEXT [options]

Options:
  --lora PATH
  --ctx-size N             Default: 8192
  --gpu-layers N           Default: 0
  --threads N              Default: 4
  --max-iterations N       Default: 256
  --no-prefix-cache        Recompute the full sequence on every D2F iteration
  --vision-gpu             Default: enabled
  --no-vision-gpu
  --preprocess-only
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$2"
            shift 2
            ;;
        --mmproj)
            mmproj="$2"
            shift 2
            ;;
        --lora)
            lora="$2"
            shift 2
            ;;
        --image)
            image="$2"
            shift 2
            ;;
        --prompt)
            prompt="$2"
            shift 2
            ;;
        --ctx-size)
            ctx_size="$2"
            shift 2
            ;;
        --gpu-layers)
            gpu_layers="$2"
            shift 2
            ;;
        --threads)
            threads="$2"
            shift 2
            ;;
        --max-iterations)
            max_iterations="$2"
            shift 2
            ;;
        --no-prefix-cache)
            prefix_cache=0
            shift
            ;;
        --vision-gpu)
            vision_gpu=1
            shift
            ;;
        --no-vision-gpu)
            vision_gpu=0
            shift
            ;;
        --preprocess-only)
            preprocess_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${model}" || -z "${mmproj}" || -z "${image}" || -z "${prompt}" ]]; then
    usage
    exit 1
fi

binary="${build_dir}/bin/llama-lladao-d2f"
backend_test="${build_dir}/bin/test-backend-ops"
scheduler_test="${build_dir}/bin/test-diffusion-d2f-scheduler"
mtmd_test="${build_dir}/bin/test-mtmd-lladao"
for path in "${binary}" "${backend_test}" "${scheduler_test}" "${mtmd_test}" "${model}" "${mmproj}" "${image}"; do
    if [[ ! -f "${path}" ]]; then
        echo "File not found: ${path}" >&2
        exit 1
    fi
done
if [[ -n "${lora}" && ! -f "${lora}" ]]; then
    echo "File not found: ${lora}" >&2
    exit 1
fi

local_size() {
    if stat -f '%z' "$1" >/dev/null 2>&1; then
        stat -f '%z' "$1"
    else
        stat -c '%s' "$1"
    fi
}

push_if_changed() {
    local source="$1"
    local destination="$2"
    local source_size
    local destination_size
    source_size="$(local_size "${source}")"
    destination_size="$("${adb_bin}" shell "stat -c '%s' '${destination}' 2>/dev/null" | tr -d '\r' || true)"
    if [[ "${source_size}" == "${destination_size}" ]]; then
        echo "Already on device: ${destination}"
        return
    fi
    "${adb_bin}" push "${source}" "${destination}"
}

quote_android() {
    local escaped="${1//\'/\'\\\'\'}"
    printf "'%s'" "${escaped}"
}

"${adb_bin}" get-state >/dev/null
"${adb_bin}" shell "mkdir -p '${remote_dir}'"

remote_binary="${remote_dir}/llama-lladao-d2f"
remote_backend_test="${remote_dir}/test-backend-ops"
remote_scheduler_test="${remote_dir}/test-diffusion-d2f-scheduler"
remote_mtmd_test="${remote_dir}/test-mtmd-lladao"
remote_model="${remote_dir}/model.gguf"
remote_mmproj="${remote_dir}/mmproj.gguf"
remote_lora="${remote_dir}/adapter.gguf"
remote_image="${remote_dir}/input-image"

"${adb_bin}" push "${binary}" "${remote_binary}"
"${adb_bin}" push "${backend_test}" "${remote_backend_test}"
"${adb_bin}" push "${scheduler_test}" "${remote_scheduler_test}"
"${adb_bin}" push "${mtmd_test}" "${remote_mtmd_test}"
push_if_changed "${model}" "${remote_model}"
push_if_changed "${mmproj}" "${remote_mmproj}"
push_if_changed "${image}" "${remote_image}"
if [[ -n "${lora}" ]]; then
    push_if_changed "${lora}" "${remote_lora}"
fi

"${adb_bin}" shell "chmod 755 '${remote_binary}' '${remote_backend_test}' '${remote_scheduler_test}' '${remote_mtmd_test}'"
"${adb_bin}" shell "'${remote_scheduler_test}' && '${remote_mtmd_test}'"
"${adb_bin}" shell "${remote_backend_test} test -b Vulkan0 -o MUL_MAT -p 'type_a=f16,type_b=f32,m=16,n=1,k=256,bs=\\[1,1\\],nr=\\[1,1\\],per=\\[0,1,2,3\\],k_v=0,o=1' -j 1"

command="$(quote_android "${remote_binary}")"
for argument in \
    --model "${remote_model}" \
    --mmproj "${remote_mmproj}" \
    --image "${remote_image}" \
    --prompt "${prompt}" \
    --ctx-size "${ctx_size}" \
    --gpu-layers "${gpu_layers}" \
    --threads "${threads}" \
    --max-iterations "${max_iterations}"; do
    command+=" $(quote_android "${argument}")"
done
if [[ -n "${lora}" ]]; then
    command+=" $(quote_android "--lora") $(quote_android "${remote_lora}")"
fi
if [[ "${vision_gpu}" -eq 1 ]]; then
    command+=" $(quote_android "--vision-gpu")"
else
    command+=" $(quote_android "--no-vision-gpu")"
fi
if [[ "${preprocess_only}" -eq 1 ]]; then
    command+=" $(quote_android "--preprocess-only")"
fi
if [[ "${prefix_cache}" -eq 0 ]]; then
    command+=" $(quote_android "--no-prefix-cache")"
fi

echo "Running on Android: ${command}"
"${adb_bin}" shell "${command}"
