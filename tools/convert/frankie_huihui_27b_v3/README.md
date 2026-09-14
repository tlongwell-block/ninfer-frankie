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
