import unittest

import torch

from conversion import MMPROJ_MODEL_MAP, get_model_class
from conversion.lladao_vision import LLaDAOVisionModel
from gguf import MODEL_ARCH, get_tensor_name_map


class TestLLaDAOVisionConverter(unittest.TestCase):
    def setUp(self):
        self.tensor_map = get_tensor_name_map(MODEL_ARCH.MMPROJ, 26)

    @staticmethod
    def normalize_name(name: str) -> str:
        item = LLaDAOVisionModel.filter_tensors((name, lambda: None))
        assert item is not None
        return item[0]

    def map_name(self, name: str) -> str | None:
        normalized = self.normalize_name(name)
        return self.tensor_map.get_name(normalized, try_suffixes=(".weight", ".bias"))

    def test_registry(self):
        self.assertEqual(MMPROJ_MODEL_MAP["LLaDAOGuiForDiffusionLM"], "lladao_vision")
        self.assertIs(get_model_class("LLaDAOGuiForDiffusionLM", mmproj=True), LLaDAOVisionModel)

    def test_special_tensor_names(self):
        expected = {
            "connector.fc1.weight": "mm.0.weight",
            "connector.fc1.bias": "mm.0.bias",
            "connector.fc2.weight": "mm.2.weight",
            "connector.fc2.bias": "mm.2.bias",
            "vit_model.vision_model.embeddings.patch_embedding.weight": "v.patch_embd.weight",
            "vit_model.vision_model.embeddings.position_embedding.weight": "v.position_embd.weight",
            "vit_model.vision_model.post_layernorm.weight": "v.post_ln.weight",
            "vit_pos_embed.pos_embed": "mm.position_embd",
        }
        self.assertEqual({name: self.map_name(name) for name in expected}, expected)

    def test_all_checkpoint_tensor_names_map(self):
        names = [
            "connector.fc1.bias",
            "connector.fc1.weight",
            "connector.fc2.bias",
            "connector.fc2.weight",
            "vit_model.vision_model.embeddings.patch_embedding.bias",
            "vit_model.vision_model.embeddings.patch_embedding.weight",
            "vit_model.vision_model.embeddings.position_embedding.weight",
            "vit_model.vision_model.post_layernorm.bias",
            "vit_model.vision_model.post_layernorm.weight",
            "vit_pos_embed.pos_embed",
        ]

        for bid in range(26):
            prefix = f"vit_model.vision_model.encoder.layers.{bid}"
            for layer_name in ("layer_norm1", "layer_norm2"):
                for suffix in ("bias", "weight"):
                    names.append(f"{prefix}.{layer_name}.{suffix}")
            for layer_name in ("mlp.fc1", "mlp.fc2"):
                for suffix in ("bias", "weight"):
                    names.append(f"{prefix}.{layer_name}.{suffix}")
            for layer_name in ("self_attn.k_proj", "self_attn.out_proj", "self_attn.q_proj", "self_attn.v_proj"):
                for suffix in ("bias", "weight"):
                    names.append(f"{prefix}.{layer_name}.{suffix}")

        self.assertEqual(len(names), 426)
        mapped = [self.map_name(name) for name in names]
        unmapped = [name for name, mapped_name in zip(names, mapped) if mapped_name is None]
        self.assertEqual(unmapped, [])
        self.assertEqual(len(set(mapped)), 426)

    def test_patch_embedding_is_restored_to_conv2d_layout(self):
        model = object.__new__(LLaDAOVisionModel)
        model.hparams_vision = {"patch_size": 2, "num_channels": 3}
        model.tensor_map = get_tensor_name_map(MODEL_ARCH.MMPROJ, 1)
        model.fuse_gate_up_exps = False

        source = torch.arange(48).reshape(4, 12)
        converted = list(model.modify_tensors(
            source,
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            None,
        ))

        self.assertEqual(len(converted), 1)
        name, tensor = converted[0]
        self.assertEqual(name, "v.patch_embd.weight")
        self.assertEqual(tuple(tensor.shape), (4, 3, 2, 2))
        torch.testing.assert_close(tensor, source.view(4, 2, 2, 3).permute(0, 3, 1, 2))


if __name__ == "__main__":
    unittest.main()
