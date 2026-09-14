"""Frankie's checkpoint identity using the existing Qwen3.8 physical contract."""

from tools.convert.qwen3_8_27b import inventory as qwen

MODEL_ID = "frankie-huihui-27b-v3"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "frankie_huihui_27b_v3"

RESOURCE_SPECS = qwen.RESOURCE_SPECS
TENSOR_SPECS = qwen.BASE_TENSOR_SPECS
OBJECT_SPECS = RESOURCE_SPECS + TENSOR_SPECS
