# Fast-dVLM

This example runs a converted Fast-dVLM multimodal checkpoint with its native
block-diffusion speculative decoder. The prompt is prefilled causally with the
Qwen2.5-VL multimodal positions. Each generation iteration then runs one
bidirectional 32-token draft and one causal verification pass, accepts the
longest matching prefix, and discards rejected KV entries.

## Convert

Convert the language model and vision projector separately, then quantize the
language GGUF if needed:

```bash
python3 convert_hf_to_gguf.py CHECKPOINT \
  --outfile fast-dvlm-f16.gguf --outtype f16
python3 convert_hf_to_gguf.py CHECKPOINT \
  --mmproj --outfile fast-dvlm-mmproj-q8_0.gguf --outtype q8_0
llama-quantize --output-tensor-type q4_0 \
  fast-dvlm-f16.gguf fast-dvlm-q4_0.gguf Q4_0
```

The converter preserves the checkpoint's image pixel limits in the mmproj
GGUF. This is important for matching the training processor and avoiding an
unintended increase in visual tokens.

## Run

```bash
llama-fast-dvlm \
  -m fast-dvlm-q4_0.gguf \
  --mmproj fast-dvlm-mmproj-q8_0.gguf \
  --image screenshot.png \
  -p 'Describe the next UI action.' \
  -c 4096 -n 96 -fa on
```

The CLI prints the generated text between `FAST_DVLM_OUTPUT_BEGIN` and
`FAST_DVLM_OUTPUT_END`. `FAST_DVLM_TIMINGS` reports model loading, image
encoding, language prefill, draft, and verification separately.

## Hexagon placement

On Snapdragon, Q4_0 language weights can be offloaded to Hexagon while keeping
the Qwen2.5-VL vision tower on CPU:

```bash
GGML_HEXAGON_NDEV=2 llama-fast-dvlm \
  -m fast-dvlm-q4_0.gguf \
  --mmproj fast-dvlm-mmproj-q8_0.gguf \
  --no-mmproj-offload \
  --device HTP0,HTP1 -ngl 999 \
  --image screenshot.png -p 'Describe the next UI action.' \
  -c 4096 -n 96 -fa on
```

Do not offload the current Q8_0 vision projector to Hexagon without checking
the graph split diagnostics. Its F32 matrix multiplications are not supported
by the current Hexagon backend and can cause frequent CPU/HTP transfers.
