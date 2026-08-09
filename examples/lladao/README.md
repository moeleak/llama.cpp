# LLaDA-o D2F inference

`llama-lladao-d2f` is the native llama.cpp inference path for the GUI-grounding
LLaDA-o checkpoint. It loads:

- the language-model GGUF;
- the LLaDA-o vision `mmproj` GGUF through `mtmd`;
- an optional F32 D2F LoRA adapter.

The runner uses the checkpoint-native resized screenshot preprocessing, the
training-time GUI instruction without a chat wrapper, native shared language
RoPE position for all tokens in one image, a 64-token output, and 16-token D2F
blocks. By default, it prefills the fixed multimodal-and-prompt prefix once per
request and reuses its KV cache. Each D2F iteration invalidates and recomputes
only the changing generation range. The cache-aware attention mask preserves
the same bidirectional image/prompt and blockwise generation visibility as the
full-sequence graph.

Build with tools, examples, and the diffusion model support enabled:

```sh
cmake -B build \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --target llama-lladao-d2f -j
```

Run a click-grounding instruction:

```sh
./build/bin/llama-lladao-d2f \
  --model /path/to/lladao-language-bf16.gguf \
  --mmproj /path/to/lladao-mmproj-bf16.gguf \
  --lora /path/to/lladao-d2f-lora-f32.gguf \
  --image /path/to/screenshot.png \
  --prompt 'Click on Track & Field.' \
  --gpu-layers 999
```

`--prompt` is the exact GUI action instruction. Do not add a system prompt,
chat template, image placeholder, or an extra question wrapper. Other examples
are `Hover over Settings.` and `Type "hello" into Search.`.

`--ctx-size` is a request limit, not an eagerly allocated cache size. The
runner sizes the actual context and KV cache to the resident token count of the
current request, so a native-resized 2K-token request does not allocate a 16K
cache. Use `--no-prefix-cache` only for a controlled performance or logits
comparison against full-sequence recomputation.

On a Pixel 9 Pro XL with the 8B Q3_K_M language model, Q8_0 vision projector,
full Vulkan language offload, and a 2,178-token native prefix, the exact F16
cache used 1,152 MiB. One measured request spent 505.17 seconds on the one-time
prefix prefill, then 10.27--12.23 seconds on each 16- or 32-token D2F pass. It
finished 15 passes in 675.49 seconds. A controlled `--no-prefix-cache` pass on
the same input recomputed 2,194 tokens in 490.10 seconds, versus 10.27 seconds
for the cached 16-token pass, a 47.7x steady-state speedup. Both passes made the
same three scheduler updates. These numbers are a device-specific smoke
benchmark, not an accuracy benchmark; the remaining one-time prefill is still
too slow for interactive use.

## Prefill/decode separation

The exact prefix KV state can cross a process boundary. The P process loads
the language model, vision projector, image, and optional LoRA, then exits
after writing the occupied prefix KV cells:

```sh
./build/bin/llama-lladao-d2f \
  --model /path/to/lladao-language-Q3_K_M.gguf \
  --mmproj /path/to/lladao-mmproj-q8_0.gguf \
  --image /path/to/screenshot.png \
  --prompt 'Click on Settings.' \
  --pd-prefill-out /path/to/prefix.state \
  --gpu-layers 999
```

The D process restores that state and runs D2F without loading `mmproj`, the
source image, or its text prompt:

```sh
./build/bin/llama-lladao-d2f \
  --model /path/to/lladao-language-Q3_K_M.gguf \
  --pd-decode-in /path/to/prefix.state \
  --gpu-layers 999
```

P and D must use the same language GGUF, RoPE settings, mask token, and LoRA
configuration. The state carries a versioned contract and rejects mismatched
model dimensions, model size, parameter count, RoPE settings, LoRA presence,
scale, or file size before decoding. This is an exact F16 KV handoff, not KV
compression. File transfer can be expensive, so local sequential PD is a
correctness and deployment primitive rather than a single-request speedup. It
becomes useful when P and D are independently scheduled or one prefetched
state is reused by multiple decode runs.

On the same Pixel 9 Pro XL configuration above, the P process prefetched 2,178
tokens in 530.13 seconds and wrote a 1,141,926,288-byte state in 3.24 seconds.
A fresh D process restored it in 2.61 seconds, completed all 15 D2F passes in
176.14 seconds, and produced the same output and per-pass update counts as the
in-process prefix cache. Observed D-process RSS was about 5.7 GiB versus about
6.4 GiB while the vision model was resident in P. This is a one-request smoke
comparison, not a 100-sample accuracy benchmark.

## Full-page tile retrieval

Use full-page mode when the original screenshot is too large for the native
single-image resize. The runner:

1. splits the original page into exact, non-overlapping 980-pixel tiles in
   row-major order;
2. pads only the right and bottom tile edges to the 14-pixel patch boundary,
   without resizing those tiles;
3. appends the checkpoint-native resized whole-page overview;
4. scores every source tile with the operation-only query using two
   complementary masked, fully bidirectional passes;
5. keeps the Top-K complete source image spans and always keeps the overview.

`--prompt` remains operation-only. The runner constructs the full tile and
overview grounding wrapper itself. `--retrieval-query` is only an optional
ablation override; by default the stripped `--prompt` text is the retrieval
query too.

```sh
./build/bin/llama-lladao-d2f \
  --model /path/to/lladao-language-bf16.gguf \
  --mmproj /path/to/lladao-mmproj-bf16.gguf \
  --lora /path/to/lladao-d2f-lora-f32.gguf \
  --image /path/to/original-full-page.png \
  --prompt 'Click on Quick Tools.' \
  --full-page-tiles \
  --full-page-tile-size 980 \
  --tile-retrieval-topk 4 \
  --tile-retrieval-mask-rounds 2 \
  --ctx-size 65536 \
  --yarn-factor 8 \
  --yarn-orig-ctx 16384 \
  --gpu-layers 999
```

This path is visual span filtering, not lossy KV-cache compression. The
retained visual spans are prefetched into the exact prefix cache, while the
discarded spans allocate no resident KV entries. The logged
`image_token_ratio` describes the retained visual token count; it is not by
itself a direct GPU-memory reduction measurement.

Use `--preprocess-only` to validate model loading, image preprocessing, token
layout, and positions without running the language model.

The initial validation target is BF16 language and vision GGUF plus an optional
F32 LoRA. Quantization does not require another LLaDA-o model implementation:
the inference graph, multimodal layout, and D2F scheduler remain the same.
Quantized language and vision tensor types still need accuracy and performance
validation, and the LoRA may either stay F32 or be merged before quantization.

## Android Vulkan

The Android target is the same native runner cross-compiled with the Android
NDK. It is an adb CLI target, so the model path can be validated independently
from an Android UI or JNI layer.

Build the arm64 binary:

```sh
export ANDROID_NDK=/path/to/android-ndk
export VULKAN_INCLUDE_DIR=/path/to/Vulkan-Headers/include
export LLADAO_CMAKE_PREFIX_PATH=/path/to/spirv-headers/install
examples/lladao/android/build-vulkan.sh
```

The NDK includes the Vulkan C API, but the backend also uses `vulkan.hpp` from
Khronos Vulkan-Headers. Set `VULKAN_INCLUDE_DIR` unless the selected NDK include
directory already provides both headers. `LLADAO_CMAKE_PREFIX_PATH` is only
needed when SPIRV-Headers is installed outside the default CMake search paths.
Cross-compiling may require the exact package directory instead:

```sh
export LLADAO_SPIRV_HEADERS_DIR=/path/to/spirv-headers/install/share/cmake/SPIRV-Headers
```

Deploy the binary, tests, model files, and one screenshot to a connected
device, then run a vision-GPU preprocessing smoke test:

```sh
examples/lladao/android/run-vulkan.sh \
  --model /path/to/lladao-language-bf16.gguf \
  --mmproj /path/to/lladao-mmproj-bf16.gguf \
  --image /path/to/screenshot.png \
  --prompt 'Click on Settings.' \
  --gpu-layers 0 \
  --vision-gpu \
  --preprocess-only
```

`--vision-gpu` is independent from `--gpu-layers`. This is useful on devices
whose Vulkan heap can hold the vision projector but not the language model.
Use `--no-vision-gpu` to validate the same path on the CPU.

The Android helper accepts `--pd-prefill-out` and `--pd-decode-in` as paths on
the device. For example, first create the state and then start a fresh D
process without passing an image or `mmproj`:

```sh
examples/lladao/android/run-vulkan.sh \
  --model /path/to/lladao-language-Q3_K_M.gguf \
  --mmproj /path/to/lladao-mmproj-q8_0.gguf \
  --image /path/to/screenshot.png \
  --prompt 'Click on Settings.' \
  --pd-prefill-out /data/local/tmp/lladao-vulkan/prefix.state \
  --gpu-layers 999

examples/lladao/android/run-vulkan.sh \
  --model /path/to/lladao-language-Q3_K_M.gguf \
  --pd-decode-in /data/local/tmp/lladao-vulkan/prefix.state \
  --gpu-layers 999
```

The unquantized language model, vision projector, and F32 adapter use about
15.94 GiB before compute buffers. The prefix cache adds storage proportional
to the request's resident token count rather than the configured maximum
context. The Android emulator with a 2 GiB Vulkan heap still cannot run the
complete model. It can validate the arm64 binary, Vulkan backend, scheduler,
multimodal loader, and vision preprocessing. Full end-to-end inference needs a
device with enough memory and a validated quantized model. This memory limit
does not require another LLaDA-o architecture port.
