"""Verify exact calibrated FP8/NVFP4 words and the original Frankie source roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.artifact.container import Artifact
from tools.artifact.layouts import decode_direct, decode_fp8_row_scaled_words, decode_nvfp4_words
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6_27b import recipe as original_recipe, verify as oracle
from tools.convert.qwen3_8_27b import fp8_embedding, recipe_nvfp4 as recipe

from . import convert, convert_nvfp4


def exact(actual, expected, name):
    if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8), expected.contiguous().reshape(-1).view(torch.uint8)):
        raise ValueError(f"{name}: stored words differ from source")


def verify(path: Path, model: Path, calibrated: Path) -> dict:
    checked = convert_nvfp4.preflight(model, calibrated)
    convert.validate_artifact(path, checked, weights_id=convert_nvfp4.WEIGHTS_ID)
    with Artifact(path) as artifact, ShardReader(model) as original, ShardReader(calibrated) as quantized:
        for selected in recipe.FP8_WEIGHT_RECIPES:
            obj = artifact.find(selected.object_name)
            codes, scales = recipe.materialize_fp8_weight(selected, quantized)
            stored_codes, stored_scales = decode_fp8_row_scaled_words(artifact.payload(obj), obj.shape)
            exact(stored_codes, codes, obj.name)
            exact(stored_scales, scales, obj.name + " scales")
        for selected in recipe.NVFP4_WEIGHT_RECIPES:
            obj = artifact.find(selected.object_name)
            packed, scales, divisor = recipe.materialize_nvfp4_weight(selected, quantized)
            stored_packed, stored_scales, stored_divisor = decode_nvfp4_words(artifact.payload(obj), obj.shape)
            exact(stored_packed, packed, obj.name)
            exact(stored_scales, scales, obj.name + " scales")
            if stored_divisor.reshape(1).view(torch.uint8).numpy().tobytes() != divisor:
                raise ValueError(f"{obj.name}: weight divisor differs from source")
        for selected in recipe.INPUT_DIVISOR_RECIPES:
            obj = artifact.find(selected.object_name)
            expected = recipe.materialize_input_divisor(selected, quantized)
            stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
            if not torch.isfinite(stored).all() or not (stored > 0).all():
                raise ValueError(f"{obj.name}: input divisor is not finite and positive")
            exact(stored, expected, obj.name)
        for spec in recipe.QUANTIZED_DIRECT_SPECS:
            obj = artifact.find(spec.name)
            expected = convert_nvfp4.profile._materialize_direct(spec, original)
            calibrated_value = convert_nvfp4.profile._materialize_direct(spec, quantized)
            exact(calibrated_value, expected, obj.name + " unchanged during calibration")
            exact(decode_direct(artifact.payload(obj), obj.format, obj.shape), expected, obj.name)
        # Independently unpack original-source MTP/vision/draft quantization rows.
        draft_ids = oracle._load_and_validate_draft_ids(artifact, model)
        slices = oracle._SourceSlices(original)
        probes = ("vision/patch_embedding", "mtp/layer/attention/query_key_gate_value",
                  "mtp/input_projection", "mtp/layer/mlp/gate_up", "text/draft_head")
        row_count = group_count = 0
        for name in probes:
            obj = artifact.find(name)
            expression = original_recipe.RECIPES_BY_NAME[name].expression
            rows = sorted({0, obj.shape[0] // 2, obj.shape[0] - 1})
            expected = oracle._materialize_rows(expression, rows, slices, draft_ids)
            group_count += oracle.verify_quantized_rows(artifact.payload(obj), obj.format, obj.shape, rows, expected, "cpu")
            row_count += len(rows)
        # The embedding uses the stock canonical FP8 encoder, not calibration.
        obj = artifact.find("text/token_embedding")
        codes, scales = decode_fp8_row_scaled_words(artifact.payload(obj), obj.shape)
        expression = recipe.OFFICIAL_EMBEDDING_SOURCE
        rows = sorted({0, obj.shape[0] // 2, obj.shape[0] - 1})
        values = oracle._materialize_rows(expression, rows, slices, draft_ids)
        expected = fp8_embedding.quantize_bf16_rows(values)
        exact(codes[rows], expected.codes, obj.name)
        exact(scales[rows], expected.scales, obj.name + " scales")
    return {"identity": {"model_id": convert.inventory.MODEL_ID, "weights_id": convert_nvfp4.WEIGHTS_ID},
            "complete_object_plan": "passed", "objects": len(checked.object_plan.objects),
            "frontend_resources_exact": len(checked.resources),
            "fp8_weights_exact": len(recipe.FP8_WEIGHT_RECIPES),
            "nvfp4_weights_exact": len(recipe.NVFP4_WEIGHT_RECIPES),
            "input_divisors_exact": len(recipe.INPUT_DIVISOR_RECIPES),
            "direct_parameters_unchanged": len(recipe.QUANTIZED_DIRECT_SPECS),
            "original_source_probes": list(probes), "quantized_rows": row_count,
            "quantized_groups": group_count, "draft_ids_exact": len(draft_ids),
            "embedding_rows_exact": len(rows), "calibration": checked.calibration,
            "cuda_inference": "not tested", "multimodal_quality": "not qualified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibrated-model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    result = verify(args.artifact, args.model, args.calibrated_model)
    Path(str(args.artifact) + ".verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
