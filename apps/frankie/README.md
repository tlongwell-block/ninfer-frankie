# Frankie Breeze

`ninfer-frankie` serves text, images, tools, and duplex voice through one process.
The existing HTTP service and `/v1/realtime` share one NInfer Engine and its loaded
brain. The Frankie runtime supplies the neural audio encoder, turn prediction,
listener reactions, expression head, and Breeze speech synthesis. It loads only
the tokenizer from the speech package's brain component.

This integration is experimental. The groupwise brain profile passes focused
RTX 5090 tests for the combined server, including custom WAV voices, neural audio
input, images, tools, interruption recovery, and concurrent HTTP progress during
speech. The reused speech runtime also passes native Metal tests.
Upstream NInfer performance figures do not measure this integration.

## Build

Use NInfer's Linux/RTX 5090 build prerequisites from the repository README, plus
the ICU development package and NVTX3 headers available to the compiler. The
reusable speech runtime is pinned in the
`frankie/ninfer-runtime` branch of the Frankie llama.cpp fork (commit
`9860bd25c0d29ce8b27fd9f311a2bce2012c1198`):

```sh
git clone --branch frankie/breeze-v3 https://github.com/tlongwell-block/ninfer-frankie.git
git clone --branch frankie/ninfer-runtime https://github.com/tlongwell-block/llama.cpp.git frankie-llama
git -C frankie-llama checkout 9860bd25c0d29ce8b27fd9f311a2bce2012c1198
cd ninfer-frankie
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_FRANKIE=ON \
  -DNINFER_LLAMA_SOURCE_DIR="$(realpath ../frankie-llama)"
cmake --build build -j --target ninfer-frankie
```

The speech runtime is linked as a library. It does not start another server or
load a second brain. Ordinary `ninfer` and `ninfer-serve` builds remain available
without enabling `NINFER_BUILD_FRANKIE`.

## Model files

Use a matching Frankie Breeze v3 brain and speech package. The
[conversion guide](../../tools/convert/frankie_huihui_27b_v3/README.md) describes
conversion from the pinned high precision checkpoint, numerical verification,
and extraction of a compact speech package from an existing full Frankie GGUF.
The compact package preserves speech tensors and the embedded reference voice;
it contains no duplicate brain or vision tensors. Neither model weights nor
voice recordings are included in this repository.

## Start the server

Start with modest context allocations for the first CUDA qualification:

```sh
./build/apps/ninfer-frankie /path/to/frankie.ninfer \
  --frankie-package /path/to/frankie-speech.gguf \
  --host 0.0.0.0 --port 8080 \
  --max-context 32768 --kv-capacity 65536 --max-concurrency 2 \
  --kv-dtype nvfp4 --spec mtp --draft-tokens 3 --lm-head-draft \
  --voice-thinking none
```

`--kv-dtype nvfp4` selects KV cache storage, independently of the brain artifact's
weight quantization. `--max-context` limits each request; `--kv-capacity` is the
shared pool, capped by `max-context × max-concurrency`. Concurrent requests need
enough room for their combined prompt and output reservations. Increase the pool
after measuring speech and graph memory.
The combined server uses CUDA device 0; it rejects other `--device` indices
to keep the brain and speech on the same GPU.
Automatic KV allocation is disabled for this executable because it would consume
memory before the speech components load.

HTTP thinking is enabled by default; `--no-thinking` disables its default.
`--voice-thinking` independently sets the realtime default. Supported voice
levels are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`.
Realtime clients can override it using `session.reasoning.effort`.
MTP settings apply to both paths. The speech prosody cache defaults to 100 words;
`--speech-context-words 0` disables it.

Supply your own voice reference with `--voice /path/to/reference.wav`. Add
`--voice-text-file /path/to/transcript.txt` if you already have its transcript;
otherwise the native audio encoder transcribes it at startup. The original packaged
reference remains available by omitting these options. `--speech-device cpu`
runs the speech components on CPU; `--text-encoder-device cpu` moves only
the Breeze text encoder to CPU while GPU speech remains enabled.

Authentication is off unless `--api-key` is supplied. When set, the same bearer
token protects HTTP and WebSocket routes. `--cors` enables browser HTTP access.
Only one client owns the realtime conversation at a time. Realtime and HTTP
share the configured active-request capacity. A voice request can queue if HTTP
already occupies every lane; there is no reserved voice lane or request
preemption. Choose enough lanes and KV capacity for the intended simultaneous
workload.

## Connect clients

Use `http://HOST:8080/v1` as the OpenAI HTTP base URL. Retrieve the model ID from
`GET /v1/models`. Existing NInfer `/v1/chat/completions`, `/v1/responses`, and
Anthropic `/v1/messages` behavior is preserved, including image and tool input.

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"frankie-huihui-27b-v3","messages":[{"role":"user","content":"Say hello briefly."}],"max_tokens":64,"stream":true}'
```

Realtime clients connect to `ws://HOST:8080/v1/realtime`. The native Frankie
protocol supports OpenAI-style session updates, conversation items, function
calls and results, streamed input/output PCM, response cancellation, and
conversation truncation. Audio is mono PCM16 at 24 kHz. It uses WebSockets;
WebRTC and SIP transport are not provided by this executable.

A minimal session update for manual push to talk is:

```json
{"type":"session.update","session":{"type":"realtime","output_modalities":["audio"],"audio":{"input":{"turn_detection":null}},"reasoning":{"effort":"none"}}}
```

Append microphone audio with `input_audio_buffer.append`, commit it with
`input_audio_buffer.commit`, then send `response.create`. During playback the
client can continue uploading audio and use `response.cancel` with
`conversation.item.truncate` to interrupt. Clients execute requested tools and
return `function_call_output` items before requesting the next response.

## Validation

Enable `BUILD_TESTING=ON` for the cache lifecycle, command-line, and voice
token/feature alignment tests. Conversion verification checks source hashes,
tensor layouts, quantized values against an independent numerical oracle, and
the absence of duplicate brain weights in the speech package.

The end-to-end harness needs Python with `websockets` and Pillow, and a mono
PCM16 24 kHz WAV asking what two plus two is. It writes recordings and a JSON
report to the selected output directory:

```sh
python apps/frankie/smoke.py --url http://localhost:8080 \
  --audio /path/to/arithmetic.wav --output /path/to/results/smoke.json
```

Set `FRANKIE_API_KEY` when the server requires authentication. The harness
executes actual dictionary lookups for HTTP and realtime tool requests and
checks tool results, unique IDs, conversation memory, images, audio input,
prosody, interruption recovery, and HTTP progress during speech.

Before calling a CUDA build validated, exercise text and images, speech input
and output, a real tool/result round trip in both protocols, interruption and
reconnect, and overlapping HTTP and realtime requests. Measure first audio,
decode throughput, MTP acceptance, and memory with speech alone and with HTTP
load. A host syntax check cannot establish GPU numerical correctness or latency.
