import unittest

import torch

from conversion import MMPROJ_MODEL_MAP, TEXT_MODEL_MAP, get_model_class
from conversion.qwenvl import Qwen2VLModel, Qwen2VLVisionModel


class TestFastDVLMConverter(unittest.TestCase):
    architecture = "Fast_dVLMForConditionalGeneration"

    def test_registry_uses_qwen2vl_text_and_vision_converters(self):
        self.assertEqual(TEXT_MODEL_MAP[self.architecture], "qwenvl")
        self.assertEqual(MMPROJ_MODEL_MAP[self.architecture], "qwenvl")
        self.assertIs(get_model_class(self.architecture), Qwen2VLModel)
        self.assertIs(get_model_class(self.architecture, mmproj=True), Qwen2VLVisionModel)

    def test_language_prefix_is_normalized(self):
        item = Qwen2VLModel.filter_tensors((
            "model.language_model.layers.3.self_attn.q_proj.weight",
            lambda: torch.empty(0),
        ))
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item[0], "model.layers.3.self_attn.q_proj.weight")

    def test_vision_prefix_is_normalized(self):
        item = Qwen2VLVisionModel.filter_tensors((
            "model.visual.blocks.7.attn.qkv.weight",
            lambda: torch.empty(0),
        ))
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item[0], "visual.blocks.7.attn.qkv.weight")

    def test_non_vision_tensor_is_excluded_from_mmproj(self):
        item = Qwen2VLVisionModel.filter_tensors((
            "model.language_model.layers.0.mlp.down_proj.weight",
            lambda: torch.empty(0),
        ))
        self.assertIsNone(item)


if __name__ == "__main__":
    unittest.main()
