from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, MmprojModel, ModelBase, gguf


@ModelBase.register("LLaDAOGuiForDiffusionLM")
class LLaDAOVisionModel(MmprojModel):
    def get_vision_config(self) -> dict[str, Any]:
        config_path = self.dir_model / "vision_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"LLaDA-o vision config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.setdefault("hidden_act", "gelu_pytorch_tanh")
        config.setdefault("layer_norm_eps", 1e-6)
        config.setdefault("num_channels", 3)
        return config

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor_config.setdefault("image_mean", [0.5, 0.5, 0.5])
        self.preprocessor_config.setdefault("image_std", [0.5, 0.5, 0.5])

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        if remote_hf_model_id is not None:
            raise ValueError("LLaDA-o mmproj conversion requires a local vision.safetensors file")

        model_path = self.dir_model / "vision.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(f"LLaDA-o vision weights not found: {model_path}")

        tensors: dict[str, Callable[[], Tensor]] = {}
        with gguf.utility.SafetensorsLocal(model_path) as model_part:
            for name, data in model_part.items():
                if self.lazy:
                    data_gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
                else:
                    dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                    data_gen = lambda data=data, dtype=dtype: torch.from_numpy(data.mmap_bytes()).view(dtype).reshape(data.shape)  # noqa: E731

                if titem := self.filter_tensors((name, data_gen)):
                    tensor_name, tensor_gen = titem
                    tensors[tensor_name] = tensor_gen

        return tensors

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LLADAO)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision["layer_norm_eps"])
        self.gguf_writer.add_vision_use_gelu(True)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("vit_model.vision_model."):
            name = name.replace("vit_model.vision_model.", "vision_tower.vision_model.", 1)
        elif name.startswith("connector.fc1."):
            name = name.replace("connector.fc1.", "multi_modal_projector.linear_0.", 1)
        elif name.startswith("connector.fc2."):
            name = name.replace("connector.fc2.", "multi_modal_projector.linear_2.", 1)
        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "vision_tower.vision_model.embeddings.patch_embedding.weight":
            assert self.hparams_vision is not None
            patch_size = int(self.hparams_vision["patch_size"])
            num_channels = int(self.hparams_vision["num_channels"])
            expected_input_dim = patch_size * patch_size * num_channels
            if data_torch.ndim != 2 or data_torch.shape[1] != expected_input_dim:
                raise ValueError(f"Unexpected LLaDA-o patch embedding shape: {tuple(data_torch.shape)}")

            data_torch = data_torch.view(data_torch.shape[0], patch_size, patch_size, num_channels)
            data_torch = data_torch.permute(0, 3, 1, 2)

        yield from super().modify_tensors(data_torch, name, bid)
