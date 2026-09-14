"""Remove duplicated brain/vision weights from a companion Frankie GGUF.

All speech tensors and embedded voice assets are copied by the native packer,
without decoding or requantization. This package requires the NInfer brain;
the normal llama.cpp brain cannot run from these metadata-only components.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def build(package: Path, output: Path, llama_source: Path) -> dict[str, object]:
    sys.path.insert(0, str(llama_source / "gguf-py"))
    import gguf

    class HeaderReader(gguf.GGUFReader):
        def _build_tensors(self, *args):
            pass  # Embedded component headers have descriptors but no payload.

    def metadata(reader):
        return {name: field.contents() for name, field in reader.fields.items()
                if not name.startswith("GGUF.")}

    if output.exists():
        raise FileExistsError(output)
    base = gguf.GGUFReader(package)
    original = {tensor.name: tensor for tensor in base.tensors}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frankie-headers-", dir=output.parent) as temporary:
        directory = Path(temporary)
        replacements = []
        for name in ("brain", "vision"):
            source = directory / (name + "-original.header")
            source.write_bytes(original[f"assets.{name}.header"].data.tobytes())
            reader = HeaderReader(source)
            path = directory / (name + ".gguf")
            writer = gguf.GGUFWriter(path, reader.get_field("general.architecture").contents())
            for field in reader.fields.values():
                if field.name.startswith("GGUF.") or field.name == "general.architecture":
                    continue
                writer.add_key_value(field.name, field.contents(), field.types[0],
                                     field.types[-1] if len(field.types) > 1 else None)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            check = gguf.GGUFReader(path)
            if check.tensors or metadata(check) != metadata(reader):
                raise ValueError(f"metadata-only {name} differs from source")
            replacements.extend(("--component", f"{name}={path}"))
        subprocess.run([sys.executable, str(llama_source / "tools/frankie/pack.py"),
                        "--base", str(package), "--output", str(output), *replacements], check=True)

    result = gguf.GGUFReader(output)
    expected = {name for name in original
                if not name.startswith(("frankie.brain.", "frankie.vision."))}
    if {tensor.name for tensor in result.tensors} != expected:
        raise ValueError("speech package tensor inventory changed unexpectedly")
    retained = 0
    for tensor in result.tensors:
        if tensor.name in ("assets.brain.header", "assets.vision.header"):
            continue
        source = original[tensor.name]
        if (tensor.tensor_type != source.tensor_type or tensor.shape.tolist() != source.shape.tolist()
                or hashlib.sha256(tensor.data).digest() != hashlib.sha256(source.data).digest()):
            raise ValueError(f"retained speech payload changed: {tensor.name}")
        retained += 1
    original_metadata, output_metadata = metadata(base), metadata(result)
    for name in ("frankie.brain.size", "frankie.vision.size"):
        original_metadata.pop(name)
        output_metadata.pop(name)
    if original_metadata != output_metadata:
        raise ValueError("unrelated package metadata changed")
    report = {"source_file": package.name, "bytes": output.stat().st_size,
              "source_bytes": package.stat().st_size, "retained_payloads_exact": retained,
              "removed_tensor_prefixes": ["frankie.brain.", "frankie.vision."],
              "brain_and_vision_metadata": "exact; zero tensors",
              "standalone_llama_brain": False, "requires": "matching Frankie NInfer brain"}
    Path(str(output) + ".verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.package, args.output, args.llama_source), indent=2))


if __name__ == "__main__":
    main()
