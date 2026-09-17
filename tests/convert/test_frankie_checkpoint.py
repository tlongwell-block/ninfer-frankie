import hashlib

import pytest

from tools.convert.frankie_huihui_27b_v3 import convert, inventory
from tools.convert.qwen3_6.common.recipe import validate_recipe_coverage
from tools.convert.qwen3_6_27b import recipe
from tools.convert.qwen3_8_27b import inventory as qwen


def test_frankie_uses_registered_mtp_only_physical_contract():
    validate_recipe_coverage(recipe.RECIPE_SPECS, inventory.TENSOR_SPECS)
    assert inventory.MODEL_ID != qwen.MODEL_ID
    assert inventory.TENSOR_SPECS == qwen.BASE_TENSOR_SPECS
    assert any(t.name.startswith("mtp/") for t in inventory.TENSOR_SPECS)
    assert not any(t.name.startswith("dflash2/") for t in inventory.TENSOR_SPECS)


@pytest.fixture
def source(tmp_path, monkeypatch):
    files = {"model-00001-of-00001.safetensors": b"checkpoint", "config.json": b"{}"}
    manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()}
    monkeypatch.setattr(convert, "CHECKPOINT", {"repo": "test/source", "revision": "pinned",
                                                "files": manifest})
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    return tmp_path


def test_checkpoint_identity_is_verified(source):
    assert convert.validate_checkpoint(source)["sha256_verified"] is True


def test_same_size_wrong_weights_are_rejected(source):
    (source / "model-00001-of-00001.safetensors").write_bytes(b"different!")
    with pytest.raises(ValueError, match="hash mismatch"):
        convert.validate_checkpoint(source)


def test_extra_shard_is_rejected(source):
    (source / "unexpected.safetensors").write_bytes(b"extra")
    with pytest.raises(ValueError, match="shard inventory"):
        convert.validate_checkpoint(source)


def test_missing_resource_is_rejected(source):
    (source / "config.json").unlink()
    with pytest.raises(ValueError, match="missing or wrong-sized"):
        convert.validate_checkpoint(source)
