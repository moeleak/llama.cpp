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
request and reuses its KV cache. Each D2F iteration then recomputes the changing
generation rows and projects dense logits for those rows. The cache-aware
attention mask preserves the same bidirectional image/prompt and blockwise
generation visibility as the full-sequence graph.

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

Two exact generation optimizations are available as explicit opt-ins:

- `--generation-block-cache` retains only the contiguous, fully resolved
  leading 16-token generation blocks. A block is cached only at a block
  boundary; a newly completed block is replayed once with its final token
  values before later iterations reuse it. The active suffix is still sent
  through every transformer layer, so strict blockwise bidirectional attention
  never observes stale KV.
- `--sparse-generation-logits` requests output projection only for currently
  masked generation positions. It does not make transformer input sparse. The
  runner maps each compact output row back to the original batch token with
  `llama_get_logits_ith()` before applying the unchanged D2F scheduler.

Both switches default to disabled until the physical-Pixel and fixed
100-sample accuracy gates pass. `--no-generation-block-cache` and
`--dense-generation-logits` are compatibility rollback switches. Batch JSON
and logs expose `d2f_input_rows` (actual transformer rows, including a repeated
prefix under `--no-prefix-cache`), `d2f_active_rows` (generation active-suffix
rows), `d2f_rebuild_rows` (newly finalized generation rows replayed at a cache
transition), `d2f_logit_rows`, and `d2f_reused_input_rows` (generation rows
skipped through block reuse). The last counter is cumulative work avoided, not
the number of resident KV cells or a direct cache-memory measurement.

These additions change the C++ layouts of `d2f_engine_params` and `d2f_result`.
JNI, static-library, and shared-library users must cleanly reconfigure and
rebuild all dependents; do not combine a new header with an old library.

Prefix prefill has four explicit modes:

- `--prefix-prefill-mode exact` is the default and submits the complete image
  plus prompt prefix as one fully bidirectional batch.
- `--prefix-prefill-mode component_exact` submits each complete image span and
  then the complete prompt. It preserves exact visibility while reducing the
  largest microbatch from the full prefix to its largest component.
- `--prefix-prefill-mode component_parallel` submits all complete image spans
  together as one image-only batch, then submits the complete prompt. The D2F
  position mask keeps image spans independent while preserving the same prompt
  visibility as `exact`; this trades a larger image microbatch for one fewer
  model call per additional image. All image spans execute in one
  `llama_decode()` graph. This is dense masked execution, not a varlen or
  block-sparse attention kernel, so masked cross-span work is still present.
- `--prefix-prefill-mode packed_image --prefix-pack-size 512` splits only image
  spans into bounded chunks and still submits the complete prompt at once. It
  changes image-token visibility across chunk boundaries, so it is an opt-in
  performance/quality ablation rather than the default.

The modified Python GUI runtime's normal `_forward_image_spans()` path loops
over image spans sequentially; `component_exact` is the matching llama.cpp
mode. `component_parallel` is an additional llama.cpp one-graph optimization,
not a claim that the Python loop runs spans concurrently. The separate
FastDLLM `generate_token_ids_from_chunk_local_kv()` path also loops over local
`prefix + chunk + query` prefills before copying per-head K/V; it is a distinct
text-chunk algorithm rather than the GUI image-span path.

Every chunk boundary, token range, component index, and elapsed time is logged.
`--release-vision-after-encode` releases the native vision context after the
image embedding has been cached. A repeated request with the same non-empty
image cache key can reuse that embedding without rebuilding vision; a cache
miss or a different image mode rebuilds the vision context.

Visual KV compaction is an additional opt-in mode:

```sh
./build/bin/llama-lladao-d2f \
  --model /path/to/lladao-language-Q3_K_L.gguf \
  --mmproj /path/to/lladao-mmproj-q8_0.gguf \
  --image /path/to/screenshot.png \
  --prompt 'Open Settings' \
  --prefix-prefill-mode component_parallel \
  --vision-kv-compression
```

The scorer captures post-RoPE prompt Q and image K from the last four language
layers, scores visual patches with the last 32 prompt queries, applies 7x7
max pooling, retains the Top-20 16x16 patch tiles, and keeps 75 percent of the
candidate visual patches independently for every language layer and KV head.
Image-span boundary tokens and the complete prompt are always retained. The
runtime then compacts K and V in place and reports the dense and active prefix
lengths, patch and tile counts, selection ratio, and compaction time.

The fixed llama.cpp KV tensor is sized before dense prefill. In-place
compaction reduces the active attention length and serialized state, but does
not shrink that already allocated tensor or its RSS. F16 cache scoring uses
F32 post-RoPE K before the cache write; a quantized K cache can therefore pick
slightly different patches from a scorer that reads quantized cache K. Visual
KV compaction is currently incompatible with PD state export/import.

A three-image-span Metal smoke comparison used the same Q3_K_L language GGUF,
Q8_0 projector, F16 grounding adapter, 720x1280 screenshot, `Open Settings`
prompt, 16K context, and deterministic D2F settings. All three arms returned
the same output bytes:

| Prefix mode | Image calls | Active prefix | Prefill | KV compact | Decode | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `component_exact` | 3 | 7,612 | 35.014 s | 0 s | 3.134 s | 38.183 s |
| `component_parallel` | 1 | 7,612 | 34.205 s | 0 s | 3.167 s | 37.404 s |
| `component_parallel` + visual KV compaction | 1 | 5,734 | 37.064 s | 6.219 s | 3.313 s | 46.634 s |

The one-call parallel arm reduced prefill by 2.31 percent and total time by
2.04 percent in this smoke run. Its largest Metal compute buffer increased
from about 679 MiB to 1,403 MiB because the implementation is dense masked.
KV compaction reduced the active prefix by 24.67 percent, but its CPU scoring
and host-staged gather cost outweighed the shorter decode in this run. These
are single-run implementation measurements, not a 100-sample accuracy result.

`--flash-attn` accepts `auto`, `enabled`, or `disabled`. F16 K/V cache supports
all three settings. A Q8_0 V cache requires `--flash-attn enabled`. Logs and
batch results report both the requested and the context-resolved setting. On
one Pixel 9 Pro XL `llama-bench` p512 run, explicit disable measured 3.25 t/s
versus 2.68 t/s with explicit enable, so Flash Attention should be benchmarked
for this workload instead of assumed faster.

Language-model CPU workers can use an engine-owned ggml threadpool. For
example, `--threads 7 --cpu-mask 0xfe --cpu-strict` assigns the seven language
workers to CPUs 1 through 7. `--cpu-poll` defaults to ggml's polling level 50.
A zero mask is the default and preserves the backend-managed thread behavior.
The engine pauses its pool and restores the caller's inherited affinity after each request, so a JNI worker is not left pinned.
The caller snapshot is taken only immediately before the first language decode, after mtmd vision encoding.
The mtmd vision encoder has its own worker configuration and does not share this threadpool; logs and batch results report `vision_threadpool_shared=false` explicitly.

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

P and D must use the same language GGUF, RoPE settings, mask token, LoRA,
cache types, Flash Attention request, prefix prefill mode, and pack size. The
state carries a versioned contract and rejects mismatched model dimensions,
model size, parameter count, or runtime configuration before decoding. This is
an exact F16 KV handoff, not KV compression. File transfer can be expensive, so
local sequential PD is a correctness and deployment primitive rather than a
single-request speedup. It becomes useful when P and D are independently
scheduled or one prefetched state is reused by multiple decode runs.

Generation-block caching and sparse-logit projection are decode execution
choices and are not serialized into the prefix state. The same exact state can
therefore be restored into either A/B arm; the dense full-generation path
remains the rollback reference.

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
