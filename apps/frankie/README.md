# Frankie Breeze

`ninfer-frankie` serves text, images, tools, and duplex voice through one process.
The existing HTTP service and `/v1/realtime` share one NInfer Engine and its loaded
brain. The Frankie runtime supplies the neural audio encoder, turn prediction,
listener reactions, expression head, and Breeze speech synthesis. It loads only
the tokenizer from the speech package's brain component.

This integration is experimental. Both the groupwise and calibrated mixed NVFP4
brain profiles pass focused RTX 5090 tests for the combined server, including custom WAV voices, neural audio
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
Qualify larger pools with simultaneous requests and long audio/image inputs;
successful startup alone does not establish that speech graphs still fit.
`--device-state-slots 0` removes spare GPU cache checkpoints while keeping the
active lane states. Cached checkpoints can use the host budget instead, trading
GPU memory for transfer overhead; it does not move model computation to CPU.
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
preemption. NInfer also admits only one cold prefill at a time: a long HTTP
prefill delays a new voice brain request even when other lanes are free. Decode
that is already active can progress between prefill chunks. Choose enough lanes
and KV capacity for the intended simultaneous workload, and use
`--pending-timeout-ms 600000` with suitably long client timeouts when requests
may queue behind large cold prompts. The default queue deadline is 30 seconds;
raising it prevents premature queue errors but does not remove this scheduling
delay.

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
It also records audio chunk receipt times, maximum/P95 inter-chunk gaps and
cumulative delivery deficit. These describe delivery timing; they do not model
an individual player's buffering or establish listening quality.

### Measured RTX 5090 results

The combined server was measured with the full Breeze speech package loaded,
MTP 3, NVFP4 KV storage, three request lanes and a 131,072-token per-request
limit. The shared KV pool was 278,528 tokens for groupwise weights and 196,608
for mixed NVFP4 weights. HTTP prose benchmarks disabled thinking, used seed 123
and requested up to 256 output tokens. Rates below measure decode separately
from prefill; outputs and MTP acceptance vary between weight profiles.

| Workload | Groupwise weights | Mixed NVFP4 weights |
| --- | ---: | ---: |
| Short prompt decode | 126–127 tok/s | 124 tok/s |
| 4,099-token prompt decode | 129–130 tok/s | 136 tok/s |
| 100,925-token prompt decode | 112 tok/s | 107–108 tok/s |
| 130,725-token prompt decode | 120 tok/s | 100 tok/s |
| Two concurrent short requests, each | 94 tok/s | 125–127 tok/s |
| Cold 100,925-token prefill | 77.8 s | 51.2 s |

The mixed NVFP4 run passed all 15 end-to-end harness records. Short speech
reached first received audio in 123–138 ms; paragraph replies started in
515–517 ms while two HTTP requests made progress. Those paragraph streams had
maximum inter-chunk gaps of 121–126 ms and no positive cumulative delivery
deficit in the measured run. These are LAN receipt timings from `response.create`,
not microphone-endpoint or physical speaker latency.

Thinking-enabled HTTP tests also executed a real lookup and calculation, then
answered correctly at short and approximately 101k context. All four call IDs
were distinct, and the long tool continuations reused over 101k prompt tokens.
The 130,725-token repeated prose request reused 130,718 tokens and reached first
content in 156 ms. That initial mixed NVFP4 run also exposed an exact-replay cache
miss. The retention correction has since been qualified with the separate groupwise
pressure test below; the complete mixed NVFP4 measurement matrix has not been repeated.

The mixed NVFP4 configuration occupied approximately 29.8 GiB of total reported
GPU memory. This includes the loaded speech components and the allocated KV
pool. The mixed recipe is larger than groupwise weights and is not a 24 GiB
configuration. These focused checks do not replace broader model-quality,
long-duration conversation or workload-specific capacity evaluation.

### Repeated conversations under cache pressure

A subsequent RTX 5090 comparison used groupwise weights, all speech/vision on GPU,
MTP 3, NVFP4 KV, a 400,000-token pool, four lanes, three spare GPU state images and
eight host state images. Each build started fresh; distinct long prompts filled
both state-image pools before a new conversation began. The correction coalesces
matching message/rewrite boundaries and lets rolling checkpoints displace eligible
inactive state within the existing limits. It also skips optional shared promotion
when its immutable source fork has not yet received a model write. Endpoint reuse
also accounts for shared optional checkpoints according to their physical ownership,
including ownership transferred by pressure eviction.

| Same-configuration workload | Before correction | Corrected build |
| --- | ---: | ---: |
| Next two turns in a new 32k conversation, cached tokens | 0 / 0 | 31,985 / 32,129 |
| Those turns, client time to first content | 18.86 / 18.30 s | 1.83 / 1.82 s |
| Second identical 99,039-token request, cached tokens | 0 | 99,032 |
| That replay, client time to first content | 79.45 s | 2.47 s |

The retention/fork-corrected run completed 28 sequential requests, including real lookup/calculation
chains at 32k and 99k context, followed by four additional identical long replays.
All ten warm long replays recomputed only seven tokens and returned identical output.
Native decode stayed around 121–124 tok/s for 512-token replies. Client-visible
rates varied with delivery timing: the main warm set had a 118 tok/s median and one
80 tok/s stall; the four follow-ups measured 117–137 tok/s. These results establish
avoided repeated prefill, not a decode-kernel speedup or a uniform latency guarantee.

Those sequential timings preceded the final endpoint-ownership correction. Its real CUDA
regression fails on the prior engine and passes with shared host state both retained and
reclaimed under pressure. The final build also passes all 15 realtime harness checks,
including simultaneous HTTP tool-result continuations, spoken tools and interruption recovery.
A separate eight-request spot check on that final build preserved all four warm 99,039-token
replays at 99,032 cached tokens and seven new tokens, with identical 512-token output and
120–125 client tok/s (120–122 native). Its real thinking-enabled tool chain advanced through
99,131 and 99,279 cached tokens; returning to the earlier exact prompt also retained its cache.

Thinking settings must stay consistent when comparing cache hits: enabling the
model's default high-effort reasoning adds system instructions and changes the
prefix. Inactive histories can still be displaced under pressure, and capture may
skip when no legal repair fits; bounded cache does not retain every conversation.
Cold prefill admission and shared voice/HTTP capacity retain the limits described
above.

Before calling a CUDA build validated, exercise text and images, speech input
and output, a real tool/result round trip in both protocols, interruption and
reconnect, and overlapping HTTP and realtime requests. Measure first audio,
decode throughput, MTP acceptance, and memory with speech alone and with HTTP
load. A host syntax check cannot establish GPU numerical correctness or latency.
