"""Calibrate the pinned Frankie brain with LLM Compressor's mixed FP8/NVFP4 recipe.

Calibration uses real forward passes; input scales are never synthesized. The
matching MTP and vision remain sourced from the original checkpoint at export.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from tools.convert.qwen3_8_27b import convert_nvfp4 as profile
from .convert import validate_checkpoint


def calibration_recipe():
    from compressed_tensors.quantization import preset_name_to_scheme
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(config_groups={
        "group_0": preset_name_to_scheme("FP8_DYNAMIC", targets=profile._FP8_TARGETS),
        "group_1": preset_name_to_scheme("NVFP4", targets=profile._NVFP4_TARGETS),
    })


@contextmanager
def preserve_norm_precision(model):
    # Qwen applies its offset norm in FP32. Prevent the calibration wrapper from
    # rounding 1 + weight to BF16, then restore the unchanged source parameters.
    originals = [(module, module.weight.data) for module in model.modules()
                 if type(module).__name__ == "Qwen3_5RMSNorm"]
    try:
        for module, weight in originals:
            module.weight.data = weight.float()
        yield
    finally:
        for module, weight in originals:
            module.weight.data = weight


def calibrate(model_dir: Path, output: Path, *, device: str, samples: int, length: int,
              dataset: str, revision: str | None) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from llmcompressor import oneshot
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if output.exists():
        raise FileExistsError(f"refusing to replace an existing calibration: {output}")
    started = time.perf_counter()
    provenance = validate_checkpoint(model_dir)
    resolved_revision = HfApi().dataset_info(dataset, revision=revision).sha
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    corpus = load_dataset(dataset, split="train_sft", revision=resolved_revision, streaming=True)
    batches = []
    digest = hashlib.sha256()
    for record in corpus.take(samples):
        ids = tokenizer.apply_chat_template(record["messages"], tokenize=True, return_dict=False, enable_thinking=False,
                                            add_generation_prompt=False, truncation=True, max_length=length)
        tensor = torch.tensor([ids], dtype=torch.int64)
        digest.update(tensor.numpy().tobytes())
        batches.append({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)})
    if len(batches) != samples:
        raise ValueError("calibration dataset has fewer samples than requested")
    print(f"source verified; calibration samples={samples} tokens={sum(b['input_ids'].numel() for b in batches)} device={device}", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager")
    model.eval()
    print("original BF16 model loaded; starting real activation calibration", flush=True)
    with preserve_norm_precision(model):
        oneshot(model=model, processor=tokenizer, recipe=calibration_recipe(),
                dataset=DataLoader(batches, batch_size=None), pipeline="basic",
                num_calibration_samples=samples, max_seq_length=length)
    model.save_pretrained(output, save_compressed=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output)
    report = {"source": provenance, "dataset": {"repository": dataset, "revision": resolved_revision,
              "split": "train_sft", "selection": "first samples", "samples": samples,
              "max_sequence_length": length, "token_ids_sha256": digest.hexdigest()},
              "device": device, "pipeline": "basic", "activation_scales": "measured forward passes",
              "elapsed_seconds": time.perf_counter() - started,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "llmcompressor", "compressed-tensors")}}
    report["files"] = {}
    for path in sorted(output.iterdir()):
        if path.suffix == ".safetensors" or path.name in ("config.json", "model.safetensors.index.json"):
            with path.open("rb") as handle:
                report["files"][path.name] = {"bytes": path.stat().st_size,
                    "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
    (output / "frankie-calibration.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()
    if min(args.samples, args.max_seq_length, args.threads) < 1:
        parser.error("samples, sequence length and threads must be positive")
    torch.set_num_threads(args.threads)
    calibrate(args.model, args.output, device=args.device, samples=args.samples,
              length=args.max_seq_length, dataset=args.dataset, revision=args.dataset_revision)


if __name__ == "__main__":
    main()
