# Frankie Huihui 27B v3

This explicitly registered `frankie-huihui-27b-v3/groupwise-int` profile uses the
original BF16 checkpoint pinned in `checkpoint.json`, including its own MTP and
vision weights. It retains the original tokenizer and processor resources. It
does not use a quantized intermediate or require DFlash2. Selecting DFlash2 is
an error because this artifact contains no DFlash2 companion.

Use Python 3.11 with the repository's conversion dependencies:

```sh
python -m tools.convert.frankie_huihui_27b_v3.convert \
  --model /path/to/pinned-source --out /path/to/frankie.ninfer --device cpu
python -m tools.convert.frankie_huihui_27b_v3.verify \
  /path/to/frankie.ninfer --model /path/to/pinned-source
```

Conversion checks every pinned source file's SHA256 before writing. It reuses
the Qwen3.8 groupwise inventory and family quantizers, layouts and MTP shortlist.
Verification checks the entire directory, frontend bytes, logical bindings,
shortlist IDs and representative stored quantization scales/codes against an
independent host oracle. Conversion reports distinguish these checks from CUDA
materialization and inference, which require the supported GPU runtime.

## Experimental mixed NVFP4 calibration

The registered `frankie-huihui-27b-v3/nvfp4` profile uses the existing Qwen3.8
mixed recipe: NVFP4 for MLP layers
0–55, row-scaled FP8 for attention, GDN projections, the output head and the last
eight MLP layers. Original-source MTP and vision use the existing family formats.
This is not an all-NVFP4 model. It retains the exact checkpoint identity above.
The generated brain artifact is 21.49 GB, compared with 18.21 GB for groupwise;
either uses the same 3.93 GB speech package (decimal units). GPU memory also
depends on context, concurrency and execution buffers.

Create a separate Python 3.11 environment with the conversion dependencies and
[LLM Compressor](https://github.com/vllm-project/llm-compressor). The initial
export path was tested with LLM Compressor revision
`3ae3cb11fbf6e2958ff64133a833148d5f18eaed`, PyTorch 2.14.0, Transformers 5.17.0 and
compressed-tensors 0.18.1.a20260911. The calibration command records all installed
versions. CPU is supported; PyTorch MPS currently rejects the library's FP8
parameter initialization. CUDA calibration requires a compatible environment.

```sh
python -m pip install torch==2.14.0 transformers==5.17.0 \
  compressed-tensors==0.18.1.a20260911 \
  'git+https://github.com/vllm-project/llm-compressor.git@3ae3cb11fbf6e2958ff64133a833148d5f18eaed'
python -m tools.convert.frankie_huihui_27b_v3.calibrate_nvfp4 \
  --model /path/to/pinned-source --output /path/to/calibrated-source \
  --device cpu --samples 32 --max-seq-length 512
python -m tools.convert.frankie_huihui_27b_v3.convert_nvfp4 \
  --model /path/to/pinned-source --calibrated-model /path/to/calibrated-source \
  --out /path/to/frankie-nvfp4.ninfer
python -m tools.convert.frankie_huihui_27b_v3.verify_nvfp4 \
  /path/to/frankie-nvfp4.ninfer --model /path/to/pinned-source \
  --calibrated-model /path/to/calibrated-source
```

Calibration measures activation scales with real forward passes over the first
32 public UltraChat samples. The report pins the dataset revision and token-ID
digest. This is an initial, text-only calibration; it does not qualify speech or
image input quality. It uses plain min/max calibration, without smoothing or
modifying the original direct parameters. The verifier checks those parameters
remain exact, every compressed matrix's stored words and divisors, and original
MTP/vision/draft representative quantization rows.

Before distributing this experimental quantization, run real CUDA comparisons
for text, image and neural-ear inputs, speech quality, layer-16 features, MTP,
and context behavior. Codec and inventory verification does not establish model
quality or GPU performance. In particular, text-derived activation scales may
need additional calibration with neural-ear and image inputs.

## Speech package

Speech stays in a matching companion Frankie Breeze v3 GGUF. To avoid storing a
second brain and vision backbone, create a speech package with the native
Frankie packer (its Python dependencies include PyYAML and tqdm):

```sh
python -m tools.convert.frankie_huihui_27b_v3.speech_package \
  --package /path/to/full-frankie.gguf --output /path/to/frankie-speech.gguf \
  --llama-source /path/to/frankie-llama.cpp
```

This preserves all other tensors and embedded voice assets byte for byte and
keeps complete brain/vision metadata with zero tensors. The speech package
requires the matching NInfer brain and cannot replace a complete model for the
ordinary llama.cpp server. Original inputs are left untouched. Component
licenses and usage restrictions remain those of the source release; conversion
does not change them.
