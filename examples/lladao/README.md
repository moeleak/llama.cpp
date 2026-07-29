# LLaDA-o D2F inference

`llama-lladao-d2f` is the first unquantized llama.cpp inference path for the
GUI-grounding LLaDA-o checkpoint. It loads:

- the language-model GGUF;
- the LLaDA-o vision `mmproj` GGUF through `mtmd`;
- an optional F32 D2F LoRA adapter.

The runner uses the checkpoint-native resized screenshot preprocessing, the
training-time GUI instruction without a chat wrapper, native shared language
RoPE position for all tokens in one image, and no persistent KV cache. Each
D2F iteration recomputes the complete active sequence with a 64-token output
and 16-token blocks.

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

Use `--preprocess-only` to validate model loading, image preprocessing, token
layout, and positions without running the language model.

The initial validation target is BF16 language and vision GGUF plus an optional
F32 LoRA. Quantization does not require another LLaDA-o model implementation:
the inference graph, multimodal layout, and D2F scheduler remain the same.
Quantized language and vision tensor types still need accuracy and performance
validation, and the LoRA may either stay F32 or be merged before quantization.
