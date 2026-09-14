import pytest
import torch

pytest.importorskip("llmcompressor")

from llmcompressor.modeling.offset_norm import norm_calibration_context
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from tools.convert.frankie_huihui_27b_v3.calibrate_nvfp4 import preserve_norm_precision


def test_plain_calibration_keeps_original_offset_norm_values_and_outputs():
    model = torch.nn.Module()
    model.config = None
    model.norm = Qwen3_5RMSNorm(64).to(torch.bfloat16)
    with torch.no_grad():
        model.norm.weight.copy_(torch.linspace(-0.02, 0.02, 64))
    original = model.norm.weight.detach().clone()
    inputs = torch.linspace(-3, 3, 128).reshape(2, 64).to(torch.bfloat16)
    expected = model.norm(inputs)
    with preserve_norm_precision(model), norm_calibration_context(model):
        assert torch.equal(model.norm(inputs), expected)
    assert torch.equal(model.norm.weight, original)
    assert torch.equal(model.norm(inputs), expected)


def test_nvfp4_scalar_divisors_are_verified_by_exact_words():
    from tools.convert.frankie_huihui_27b_v3.verify_nvfp4 import exact
    value = torch.tensor(1.0, dtype=torch.float32)
    exact(value, value.clone(), "divisor")
    changed = value.clone()
    changed.view(torch.int32).add_(1)
    with pytest.raises(ValueError, match="stored words"):
        exact(value, changed, "divisor")


def test_compressed_mixed_calibration_uses_existing_ninfer_codecs(tmp_path):
    import json
    from llmcompressor import oneshot
    from safetensors import safe_open
    from tokenizers import Tokenizer, models
    from torch.utils.data import DataLoader
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tools.artifact.layouts import (
        encode_fp8_row_scaled, decode_fp8_row_scaled_words, encode_nvfp4, decode_nvfp4_words,
    )
    from tools.convert.frankie_huihui_27b_v3.calibrate_nvfp4 import calibration_recipe
    from tools.convert.qwen3_8_27b import convert_nvfp4 as profile

    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=128)).to(torch.bfloat16)
    source = tmp_path / "source"
    model.config.save_pretrained(source)
    model.config.name_or_path = str(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")), unk_token="[UNK]")
    samples = [{"input_ids": torch.randint(0, 128, (1, 32)),
                "attention_mask": torch.ones(1, 32, dtype=torch.int64)} for _ in range(2)]
    output = tmp_path / "quantized"
    with preserve_norm_precision(model):
        oneshot(model=model, processor=tokenizer, recipe=calibration_recipe(),
                dataset=DataLoader(samples, batch_size=None), pipeline="basic",
                num_calibration_samples=2, max_seq_length=32)
    model.save_pretrained(output, save_compressed=True, max_shard_size="5GB")
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["format"] == "mixed-precision" and config["quantization_status"] == "compressed"
    profile._validate_float_group(config["config_groups"]["group_0"])
    profile._validate_nvfp4_group(config["config_groups"]["group_1"])
    with safe_open(str(output / "model.safetensors"), framework="pt", device="cpu") as stored:
        name = "model.layers.0.self_attn.q_proj"
        codes = stored.get_tensor(name + ".weight").view(torch.uint8)
        scales = stored.get_tensor(name + ".weight_scale").reshape(-1)
        payload = encode_fp8_row_scaled(codes, scales, tuple(codes.shape))
        actual_codes, actual_scales = decode_fp8_row_scaled_words(payload, tuple(codes.shape))
        assert torch.equal(codes, actual_codes)
        assert torch.equal(scales.view(torch.int16), actual_scales.view(torch.int16))
        name = "model.layers.0.mlp.gate_proj"
        packed = stored.get_tensor(name + ".weight_packed")
        scales = stored.get_tensor(name + ".weight_scale").view(torch.uint8)
        divisor = stored.get_tensor(name + ".weight_global_scale").view(torch.uint8).numpy().tobytes()
        shape = (packed.shape[0], packed.shape[1] * 2)
        payload = encode_nvfp4(packed, scales, divisor, shape)
        actual_packed, actual_scales, actual_divisor = decode_nvfp4_words(payload, shape)
        assert torch.equal(packed, actual_packed)
        assert torch.equal(scales, actual_scales)
        assert actual_divisor.reshape(1).view(torch.uint8).numpy().tobytes() == divisor
        activation = stored.get_tensor(name + ".input_global_scale")
        assert torch.isfinite(activation).all() and (activation > 0).all()
        for field in ("weight_global_scale", "input_global_scale"):
            gate = stored.get_tensor("model.layers.0.mlp.gate_proj." + field)
            up = stored.get_tensor("model.layers.0.mlp.up_proj." + field)
            assert torch.equal(gate.view(torch.int32), up.view(torch.int32))
