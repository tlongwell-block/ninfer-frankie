"""Convert the exact Frankie v3 Huihui BF16 source, without DFlash2.

python -m tools.convert.frankie_huihui_27b_v3.convert \
    --model /path/to/pinned-huihui-source --out /path/to/brain.ninfer --device cpu

The registered source includes the matching MTP and vision tensors. This recipe
does not accept the MLX/GGUF quantization as an intermediate or substitute the
official Qwen brain. Speech assets remain in the companion Frankie GGUF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as base
from tools.convert.qwen3_6_27b import draft_head, recipe
from tools.convert.qwen3_8_27b import convert as qwen38

from . import inventory

RECIPE_ID = "frankie-huihui-27b-v3-groupwise-mtp-v1"
CHECKPOINT = json.loads(Path(__file__).with_name("checkpoint.json").read_text())


def validate_checkpoint(model_dir: Path) -> dict[str, object]:
    """Check source identity before writing any artifact payloads."""
    expected = CHECKPOINT["files"]
    actual_shards = {path.name for path in model_dir.glob("*.safetensors")}
    if actual_shards != {name for name in expected if name.endswith(".safetensors")}:
        raise ValueError("Frankie checkpoint shard inventory differs from the pinned source")
    for name, record in expected.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Frankie checkpoint missing or wrong-sized source: {name}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Frankie checkpoint source hash mismatch: {name}")
    return {"repository": CHECKPOINT["repo"], "revision": CHECKPOINT["revision"],
            "verified_files": len(expected), "sha256_verified": True,
            "quantized_intermediate": False, "mtp_source": "same checkpoint"}


def preflight_conversion(model_dir: Path) -> tuple[base.ConversionPreflight, dict[str, object]]:
    config = base.validate_config(family.load_json(model_dir / "config.json"))
    provenance = validate_checkpoint(model_dir)
    family_recipe.validate_recipe_coverage(recipe.RECIPE_SPECS, inventory.TENSOR_SPECS)
    source = recipe.preflight_sources(model_dir)
    # This finetune retained the official tokenizer, processor and chat template.
    # Reuse their exact existing validation, without assuming official weights.
    resources = qwen38.load_resources(model_dir)
    plan = family.build_object_plan(inventory.OBJECT_SPECS, {r.name: r.data for r in resources})
    ranking = Path(__file__).resolve().parents[3] / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model_dir)
    return base.ConversionPreflight(model_dir, config, source, resources, draft, plan), provenance


def validate_artifact(path: Path, preflight: base.ConversionPreflight, *, weights_id: str = inventory.WEIGHTS_ID) -> None:
    with Artifact(path) as artifact:
        if artifact.identity != ArtifactIdentity(inventory.MODEL_ID, weights_id):
            raise ValueError("converted Frankie artifact has the wrong checkpoint identity")
        if artifact.objects != preflight.object_plan.objects:
            raise ValueError("converted Frankie object directory differs from its complete plan")
        for resource in preflight.resources:
            if bytes(artifact.payload(resource.name)) != resource.data:
                raise ValueError(f"converted frontend resource mismatch: {resource.name}")


def convert(model_dir: Path, output: Path, *, device: str = "cpu") -> Path:
    started = time.perf_counter()
    resolved_device = pick_device(device)
    preflight, provenance = preflight_conversion(model_dir)
    print(f"verified pinned Huihui source; {len(preflight.object_plan.objects)} objects; "
          f"device={resolved_device}; native MTP, no DFlash2", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing artifact: {output}")
    resources = {resource.name: resource.data for resource in preflight.resources}
    # A partial conversion must not look like a ready-to-load model.
    partial = output.with_suffix(output.suffix + ".partial")
    with ArtifactWriter(partial, ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
                        preflight.object_plan.specs) as writer, ShardReader(model_dir) as reader:
        for index, spec in enumerate(inventory.OBJECT_SPECS, 1):
            if spec.name in resources:
                payload = resources[spec.name]
            else:
                tensor = base.materialize_tensor(spec, reader, preflight.draft)
                payload = family.encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
            writer.write(spec.name, payload)
            del payload
            print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)
    validate_artifact(partial, preflight)
    partial.rename(output)
    root = Path(__file__).resolve().parents[3]
    report = family.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY, recipe_id=RECIPE_ID, repo_root=root,
        model_dir=model_dir, out_path=output,
        arguments={"model": str(model_dir), "out": str(output), "device": device},
        config_summary=preflight.config_summary, source_preflight=preflight.source,
        objects=preflight.object_plan.objects, elapsed_seconds=time.perf_counter() - started,
        final_bytes=output.stat().st_size, device=resolved_device,
        ranking_path=root / draft_head.DEFAULT_RANKING)
    report["source"] = provenance
    report["speculation"] = {"mtp": True, "dflash2": False}
    report["validation"] = {"source_hashes": "passed", "complete_object_plan": "passed",
                            "frontend_bytes": "exact", "cuda_inference": "not tested"}
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"complete: {output.stat().st_size} bytes; report={report_path}", flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
