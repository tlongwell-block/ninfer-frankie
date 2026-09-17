"""Verify the pinned Frankie artifact using the existing independent host oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.artifact.container import Artifact
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6_27b import recipe, verify as oracle

from . import convert, inventory


def verify(path: Path, model_dir: Path) -> dict[str, object]:
    preflight, provenance = convert.preflight_conversion(model_dir)
    convert.validate_artifact(path, preflight)
    probes = (*oracle.QUANT_PROBE_OBJECTS, "text/token_embedding", "text/output_head",
              "mtp/input_projection", "mtp/layer/mlp/gate_up")
    rows_checked = groups_checked = 0
    with Artifact(path) as artifact, ShardReader(model_dir) as source:
        views, aliases = oracle.validate_logical_bindings(artifact.objects)
        draft_ids = oracle._load_and_validate_draft_ids(artifact, model_dir)
        for name in oracle.DIRECT_PROBE_OBJECTS:
            oracle._verify_direct_probe(artifact, source, name)
        slices = oracle._SourceSlices(source)
        for name in probes:
            obj = artifact.find(name)
            expression = recipe.RECIPES_BY_NAME[name].expression
            rows = {0, obj.shape[0] // 2, obj.shape[0] - 1}
            # Probe both sides of fused matrix boundaries as well as endpoints.
            if isinstance(expression, recipe.Concat) and expression.axis == 0:
                boundary = 0
                for part in expression.sources[:-1]:
                    boundary += recipe.expression_shape(part)[0]
                    rows.update((boundary - 1, boundary))
            rows = sorted(rows)
            values = oracle._materialize_rows(expression, rows, slices, draft_ids)
            groups_checked += oracle.verify_quantized_rows(
                artifact.payload(obj), obj.format, obj.shape, rows, values, "cpu")
            rows_checked += len(rows)
    return {"identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
            "source": provenance, "complete_object_plan": "passed",
            "objects": len(preflight.object_plan.objects), "logical_row_views": views,
            "logical_aliases": aliases, "frontend_resources_exact": len(preflight.resources),
            "draft_ids_exact": len(draft_ids), "direct_probes": len(oracle.DIRECT_PROBE_OBJECTS),
            "quantized_probes": list(probes), "quantized_rows": rows_checked,
            "quantized_groups": groups_checked,
            "quantization_oracle": "independent NumPy FP64/FP32; exact FP16 scales and codes",
            "cuda_inference": "not tested"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.artifact, args.model)
    output = Path(str(args.artifact) + ".verification.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
