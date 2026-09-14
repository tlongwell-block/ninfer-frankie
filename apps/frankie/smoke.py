"""Exercise a running single-process Frankie server, including live PCM ingress, concurrent HTTP decoding, and real tools."""

import argparse
import asyncio
import json
import base64
import time
import wave
import re
import os
import io
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
import websockets
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--url", default="http://127.0.0.1:8080", help="Server origin")
ap.add_argument(
    "--audio",
    type=Path,
    required=True,
    help="Mono PCM16 24 kHz WAV asking what two plus two is",
)
ap.add_argument("--output", type=Path, default=Path("frankie-results.json"))
a = ap.parse_args()
token = os.environ.get("FRANKIE_API_KEY", "")
headers = {"Authorization": "Bearer " + token} if token else {}
a.output.parent.mkdir(parents=True, exist_ok=True)
origin = a.url.rstrip("/")
parsed = urlsplit(origin)
ws_url = urlunsplit(("wss" if parsed.scheme == "https" else "ws", parsed.netloc, "/v1/realtime", "", ""))

with wave.open(str(a.audio)) as w:
    assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
    pcm = w.readframes(w.getnframes())

fixture = Image.new("RGB", (420, 180), "white")
draw = ImageDraw.Draw(fixture)
for x, color in [(20, "red"), (160, "green"), (300, "blue")]:
    draw.rectangle((x, 30, x + 100, 130), fill=color)
png = io.BytesIO()
fixture.save(png, format="PNG")
results = []


def save_results():
    a.output.write_text(json.dumps(results, indent=2))



def http_json(path, body=None, stream=False):
    request = urllib.request.Request(origin + path, headers={"Content-Type": "application/json", **headers},
                                     data=None if body is None else json.dumps(body).encode())
    timing = {"first_content_at": None, "first_tool_delta_at": None}
    with urllib.request.urlopen(request, timeout=180) as reply:
        if not stream:
            return json.load(reply), time.monotonic()
        calls, content, usage = {}, "", {}
        for raw in reply:
            if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                continue
            event = json.loads(raw[6:])
            if event.get("error"):
                raise AssertionError(event)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") and timing["first_content_at"] is None:
                    timing["first_content_at"] = time.monotonic()
                if delta.get("tool_calls") and timing["first_tool_delta_at"] is None:
                    timing["first_tool_delta_at"] = time.monotonic()
                content += delta.get("content") or ""
                for call in delta.get("tool_calls", []):
                    item = calls.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    item["id"] += call.get("id") or ""
                    for key in ("name", "arguments"):
                        item["function"][key] += call.get("function", {}).get(key) or ""
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [calls[n] for n in sorted(calls)]
        timing["finished_at"] = time.monotonic()
        return {"choices": [{"message": message}], "usage": usage}, timing


def http_tool_roundtrip(index):
    started = time.monotonic()
    result = {"name": f"http-tool-{index}", "started_at": started}
    try:
        model = http_json("/v1/models")[0]["data"][0]["id"]
        key = f"route-{index}"
        values = {"route-0": 173, "route-1": 281}
        messages = [{"role": "user", "content": f"Use lookup_route to find {key}. Report the returned number only."}]
        tools = [{"type": "function", "function": {"name": "lookup_route", "description": "Return a route number by name.",
                  "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}]
        body = {"model": model, "messages": messages, "tools": tools, "max_tokens": 256,
                "reasoning_effort": "none", "stream": True}
        first, timing = http_json("/v1/chat/completions", body, stream=True)
        result.update(timing)
        result["tool_response"] = first
        # Tool syntax can be buffered until the complete call. This measures its publication,
        # not time to the first decoded token; ordinary content streams prove overlap below.
        if timing["first_tool_delta_at"] is not None:
            result["first_tool_delta_ms"] = (timing["first_tool_delta_at"] - started) * 1000
        message = first["choices"][0]["message"]
        calls = message.get("tool_calls", [])
        assert len(calls) == 1 and calls[0]["id"], first
        call = calls[0]
        result["call_id"] = call["id"]
        assert call["function"]["name"] == "lookup_route", call
        arguments = json.loads(call["function"]["arguments"])
        assert arguments["name"] == key, arguments
        actual = values[arguments["name"]]
        messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"number": actual})}])
        body["stream"] = False
        final, _ = http_json("/v1/chat/completions", body)
        result["response"] = final
        answer = final["choices"][0]["message"]
        assert not answer.get("tool_calls") and str(actual) in answer.get("content", ""), final
    except Exception as error:
        # Return the evidence even on failure; the async caller persists every task before
        # raising, so one bad tool result cannot erase the other request or speech timing.
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = time.monotonic() - started
    return result


def http_content_stream(index):
    started = time.monotonic()
    result = {"name": f"http-content-{index}", "started_at": started}
    try:
        model = http_json("/v1/models")[0]["data"][0]["id"]
        body = {"model": model, "messages": [{"role": "user", "content":
                f"Describe a peaceful {'garden' if index == 0 else 'beach'} in four short sentences. Begin directly."}],
                "max_tokens": 128, "reasoning_effort": "none", "stream": True}
        response, timing = http_json("/v1/chat/completions", body, stream=True)
        result.update(timing)
        result["response"] = response
        if timing["first_content_at"] is not None:
            result["first_content_ms"] = (timing["first_content_at"] - started) * 1000
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = time.monotonic() - started
    return result


async def main():
    async with websockets.connect(
        ws_url,
        additional_headers=headers,
        max_size=16 * 1024**2,
    ) as ws:

        async def send(**x):
            await ws.send(json.dumps(x))

        async def until(kind):
            while True:
                e = json.loads(await asyncio.wait_for(ws.recv(), 90))
                if e["type"] == "error":
                    raise AssertionError(e)
                if e["type"] == kind:
                    return e

        async def update(**s):
            s["type"] = "realtime"
            s.pop("temperature", None)
            if "thinking" in s:
                effort = s.pop("thinking")
                s["reasoning"] = {"effort": "none" if effort == "off" else effort}
            await send(type="session.update", session=s)
            await until("session.updated")

        async def add(text=None, image=None, audio=None):
            if audio is not None:
                assert not text and not image
                for offset in range(0, len(audio), 960):
                    await send(type="input_audio_buffer.append", audio=base64.b64encode(audio[offset:offset + 960]).decode())
                    await asyncio.sleep(0.02)
                await send(type="input_audio_buffer.commit")
                await until("input_audio_buffer.committed")
                return
            c = []
            if text:
                c.append({"type": "input_text", "text": text})
            if image:
                c.append(
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(image).decode(),
                    }
                )
            if audio:
                c.append(
                    {"type": "input_audio", "audio": base64.b64encode(audio).decode()}
                )
            await send(
                type="conversation.item.create",
                item={"type": "message", "role": "user", "content": c},
            )
            await until("conversation.item.created")

        async def response(name, overlap=None):
            t = time.monotonic()
            await send(type="response.create")
            frames, events, http_tasks = [], [], []
            metrics = first = first_audio_at = last_audio_at = voice_done = r = None
            failure = None
            try:
                while True:
                    e = json.loads(await asyncio.wait_for(ws.recv(), 90))
                    events.append(e["type"])
                    if e["type"] == "error":
                        raise AssertionError(e)
                    if e["type"] == "response.output_audio.delta":
                        chunk = base64.b64decode(e["delta"])
                        if not chunk:
                            continue
                        frames.append(chunk)
                        last_audio_at = time.monotonic()
                        if first is None:
                            first_audio_at = last_audio_at
                            first = first_audio_at - t
                            if overlap:
                                work = http_content_stream if overlap == "content" else http_tool_roundtrip
                                http_tasks = [asyncio.create_task(asyncio.to_thread(work, n)) for n in range(2)]
                    if e["type"] == "frankie.metrics":
                        metrics = e["metrics"]
                    if e["type"] == "response.done":
                        r = e["response"]
                        voice_done = time.monotonic()
                        break
            except Exception as error:
                failure = error
            if frames:
                with wave.open(str(a.output.parent / f"{name}.wav"), "wb") as saved:
                    saved.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    saved.writeframes(b"".join(frames))
            row = {
                "name": name,
                "seconds": time.monotonic() - t,
                "started_at": t,
                "first_audio": first,
                "first_audio_at": first_audio_at,
                "last_audio_at": last_audio_at,
                "voice_done_at": voice_done,
                "audio_seconds": sum(map(len, frames)) / 48000,
                "response": r,
                "metrics": metrics,
                "events": events,
            }
            if overlap:
                row["overlap"] = overlap
            if failure:
                row["error"] = f"{type(failure).__name__}: {failure}"
            results.append(row)
            save_results()
            http_results = await asyncio.gather(*http_tasks, return_exceptions=True)
            for index, result in enumerate(http_results):
                if isinstance(result, BaseException):
                    result = {"name": f"http-{overlap}-{index}", "error": str(result)}
                    http_results[index] = result
                results.append(result)
            # Persist speech and every concurrent task before any overlap/correctness assertion.
            save_results()
            print(json.dumps(row), flush=True)
            if failure:
                raise failure
            assert r["status"] == "completed", r
            if overlap:
                assert len(http_results) == 2, "overlap test received no speech"
                for result in http_results:
                    assert "error" not in result, result
                    if overlap == "content":
                        first_content_at = result.get("first_content_at")
                        assert first_content_at is not None, "HTTP stream produced no content"
                        assert first_audio_at < first_content_at < last_audio_at <= voice_done, (
                            "HTTP must deliver content after speech starts and before a later audio delta", result, row)
            return r

        def text(r):
            return " ".join(
                p.get("text", p.get("transcript", ""))
                for i in r["output"]
                if i["type"] == "message"
                for p in i["content"]
            )

        await until("session.created")
        await update(
            temperature=0,
            max_output_tokens=128,
            audio={"input": {"turn_detection": None}},
            output_modalities=["text"],
        )
        await add("My checkpoint is crimson-orbit-7429. Remember it and say okay.")
        await response("text")
        await add("What is my checkpoint?")
        r = await response("memory")
        assert "7429" in text(r), r
        await add("What is in this image?", image=png.getvalue())
        r = await response("image")
        assert any(k in text(r).lower() for k in ["red", "blue"]), r
        await update(output_modalities=["audio"])
        await add("What is seven times eight? Answer briefly.")
        r = await response("speech")
        assert "56" in text(r) or "fifty" in text(r).lower(), r
        await add(audio=pcm)
        r = await response("audio-input")
        assert re.search(r"\b(four|4)\b", text(r).lower()), r
        paragraph = (
            "The little garden is quiet this morning. "
            "A warm breeze moves gently through the trees. "
            "We can sit here for a while and enjoy the sunshine. "
            "Later we will walk down the path toward the river. "
            "The water catches the light beneath the old wooden bridge. "
            "Small birds gather on the fence beside a field of flowers. "
            "We will take our time and listen to the sounds around us. "
            "There is no hurry on a morning as peaceful as this one."
        )
        await update(max_output_tokens=256)
        for overlap in ("content", "tools"):
            await add("Read this paragraph aloud exactly as written, with no introduction: " + paragraph)
            r = await response(f"prosody-paragraph-{overlap}", overlap=overlap)
            assert "peaceful" in text(r).lower(), r
        await update(
            output_modalities=["text"], thinking="minimal", max_output_tokens=256
        )
        await add("What is my checkpoint?")
        r = await response("thinking")
        assert "7429" in text(r), r
        await update(
            thinking="off", output_modalities=["audio"],
            tools=[
                {
                    "type": "function",
                    "name": "lookup_checkpoint",
                    "description": "Look up a checkpoint.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                }
            ],
        )
        await add("Call lookup_checkpoint with key lighthouse.")
        r = await response("tool-call")
        calls = [i for i in r["output"] if i["type"] == "function_call"]
        assert len(calls) == 1, r
        await send(
            type="conversation.item.create",
            item={
                "type": "function_call_output",
                "call_id": calls[0]["call_id"],
                "output": json.dumps({"code": {"lighthouse": "cedar-58"}[json.loads(calls[0]["arguments"])["key"]]}),
            },
        )
        await until("conversation.item.created")
        r = await response("tool-result")
        assert re.search(r"58|fifty[- ]eight", text(r).lower()), r
        assert results[-1]["audio_seconds"] > 0, "tool result was not spoken"
        # Keep sending real-time microphone frames while speech is being produced,
        # then interrupt and submit the captured utterance as the next turn.
        await update(
            thinking="off", tools=[], output_modalities=["audio"], max_output_tokens=512
        )
        await add("Describe a peaceful garden in ten sentences.")
        await send(type="response.create")
        first = await until("response.output_audio.delta")
        started = time.monotonic()
        for offset in range(0, len(pcm), 960):
            await send(
                type="input_audio_buffer.append",
                audio=base64.b64encode(pcm[offset : offset + 960]).decode(),
            )
            if offset == 0:
                await send(type="response.cancel", response_id=first["response_id"])
                await send(
                    type="conversation.item.truncate",
                    item_id=first["item_id"],
                    content_index=0,
                    audio_end_ms=0,
                )
            await asyncio.sleep(
                max(0, started + (offset + 960) / 48000 - time.monotonic())
            )
        cancelled = await until("response.done")
        assert cancelled["response"]["status"] == "cancelled"
        await send(type="conversation.item.retrieve", item_id=first["item_id"])
        heard = await until("conversation.item.retrieved")
        heard_text = heard["item"]["content"][0]["transcript"].strip()
        assert heard_text in {"", "[interrupted by the user]"}
        await send(type="input_audio_buffer.commit")
        await until("input_audio_buffer.committed")
        r = await response("duplex-cancel-and-continue")
        assert re.search(r"\b(four|4)\b", text(r).lower()), r
    save_results()
    ids = [r["call_id"] for r in results if r["name"].startswith("http-tool-")]
    assert len(ids) == len(set(ids)), "tool call IDs collided across requests"
    print("PASS: " + str(a.output))


try:
    asyncio.run(main())
except Exception as error:
    results.append({"name": "failure", "error": f"{type(error).__name__}: {error}"})
    save_results()
    raise
