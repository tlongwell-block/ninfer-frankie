"""Convert the genuinely calibrated Frankie checkpoint into the existing NVFP4 layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import encode_direct
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_8_27b import convert_nvfp4 as profile, fp8_embedding
from tools.convert.qwen3_8_27b import inventory_nvfp4 as inventory, recipe_nvfp4 as recipe

from . import convert as frankie

WEIGHTS_ID = "nvfp4"


def validate_calibration(directory: Path) -> dict:
    report = json.loads((directory / "frankie-calibration.json").read_text())
    if (report["source"]["repository"] != frankie.CHECKPOINT["repo"] or
            report["source"]["revision"] != frankie.CHECKPOINT["revision"] or
            report["activation_scales"] != "measured forward passes" or
            report["dataset"]["samples"] < 1):
        raise ValueError("calibration does not identify the pinned Frankie checkpoint")
    expected = report["files"]
    actual = {p.name for p in directory.glob("*.safetensors")}
    if actual != {name for name in expected if name.endswith(".safetensors")}:
        raise ValueError("calibration shard inventory changed")
    for name, record in expected.items():
        path = directory / name
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"calibration file size changed: {name}")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"calibration file hash changed: {name}")
    return report


def preflight(model: Path, calibrated: Path):
    source = frankie.validate_checkpoint(model)
    calibration = validate_calibration(calibrated)
    profile._validate_index(model)
    profile._validate_index(calibrated)
    config = profile.family_config.validate_config(family.load_json(model / "config.json"))
    if profile._validate_quantized_config(family.load_json(calibrated / "config.json")) != config:
        raise ValueError("calibrated geometry differs from original Frankie")
    with ShardReader(model) as reader:
        recipe.preflight_official_sources(reader)
    with ShardReader(calibrated) as reader:
        recipe.preflight_quantized_metadata(reader)
    resources = profile.base_convert.load_resources(model)
    objects = inventory.RESOURCE_SPECS + inventory.BASE_TENSOR_SPECS
    plan = family.build_object_plan(objects, {r.name: r.data for r in resources})
    return SimpleNamespace(resources=resources, object_plan=plan, source=source,
                           calibration=calibration, config=config)


def convert(model: Path, calibrated: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    checked = preflight(model, calibrated)
    root = Path(__file__).resolve().parents[3]
    draft = draft_head.compute_shortlist(root / draft_head.DEFAULT_RANKING, model)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_head.materialize_draft_head_token_ids(draft)}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with ArtifactWriter(partial, ArtifactIdentity(frankie.inventory.MODEL_ID, WEIGHTS_ID),
                        checked.object_plan.specs) as writer:
        for resource in checked.resources:
            writer.write(resource.name, resource.data)
        with ShardReader(model) as original, ShardReader(calibrated) as quantized:
            for index, spec in enumerate(inventory.BASE_TENSOR_SPECS, 1):
                if spec.name == "text/token_embedding":
                    payload = fp8_embedding.iter_reader_payload(original, recipe.OFFICIAL_EMBEDDING_SOURCE.name, spec.shape)
                elif spec.name in recipe.FP8_WEIGHTS_BY_NAME:
                    payload = profile._encode_fp8_weight(spec, quantized)
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = profile._encode_nvfp4_weight(spec, quantized)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(recipe.INPUT_DIVISORS_BY_NAME[spec.name], quantized)
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.QUANTIZED_DIRECT_BY_NAME:
                    # Plain calibration does not smooth or transform direct parameters.
                    payload = encode_direct(profile._materialize_direct(spec, original), spec.format)
                else:
                    tensor = profile._materialize_official(spec, original, derived)
                    payload = family.encode_tensor_payload(tensor, spec, torch.device("cpu"))
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{len(inventory.BASE_TENSOR_SPECS)}] {spec.name}", flush=True)
    frankie.validate_artifact(partial, checked, weights_id=WEIGHTS_ID)
    partial.rename(output)
    report = {"identity": {"model_id": frankie.inventory.MODEL_ID, "weights_id": WEIGHTS_ID},
              "source": checked.source, "calibration": checked.calibration,
              "speculation": {"mtp": True, "dflash2": False}, "bytes": output.stat().st_size,
              "objects": len(checked.object_plan.objects), "elapsed_seconds": time.perf_counter() - started,
              "cuda_inference": "not tested"}
    Path(str(output) + ".conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"converted {output.stat().st_size} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibrated-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    convert(args.model, args.calibrated_model, args.out)


if __name__ == "__main__":
    main()
