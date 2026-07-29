from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("LLaDAModelLM", "LLaDAOGuiForDiffusionLM")
class LLaDAModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLADA
    undo_permute = True

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item

        if "_moe_gen" in name or ".lora_A" in name or ".lora_B" in name:
            return None

        return super().filter_tensors(item)

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

        vocab_dict = tokenizer.get_vocab()  # ty: ignore[unresolved-attribute]
        vocab_size = self.hparams.get("vocab_size", len(vocab_dict))
        assert max(vocab_dict.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab_dict.items()}
        added_vocab = tokenizer.get_added_vocab()  # ty: ignore[unresolved-attribute]

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                # Check if it's a special token - treat special tokens as CONTROL tokens
                if hasattr(tokenizer, 'added_tokens_decoder') and i in tokenizer.added_tokens_decoder:
                    if tokenizer.added_tokens_decoder[i].special:
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    # Fallback: treat all added vocab as control tokens for special tokens like <|im_start|>
                    toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        if self.hf_arch == "LLaDAOGuiForDiffusionLM":
            # LLaDA-o adds these two tokens dynamically before resizing the
            # embedding table. The exported tokenizer files predate that
            # runtime mutation, so restore the exact reserved IDs here.
            for token_id, token in ((126349, "<|vision_start|>"), (126350, "<|vision_end|>")):
                if token_id >= vocab_size:
                    raise ValueError(f"LLaDA-o vocabulary is too small for {token}: {vocab_size}")
                tokens[token_id] = token
                toktypes[token_id] = gguf.TokenType.CONTROL

        return tokens, toktypes, tokpre

    def set_vocab(self):
        self._set_vocab_gpt2()

        # Legacy LLaDA checkpoints expect automatic BOS insertion. LLaDA-o
        # builds its multimodal sequence explicitly, so preserve the tokenizer
        # metadata and let the D2F runner insert BOS exactly once.
        if self.hf_arch != "LLaDAOGuiForDiffusionLM":
            self.gguf_writer.add_add_bos_token(True)

    def set_gguf_parameters(self):
        hparams = self.hparams
        if "hidden_size" not in hparams and "d_model" in hparams:
            hparams["hidden_size"] = hparams["d_model"]
        if "intermediate_size" not in hparams and "mlp_hidden_size" in hparams:
            hparams["intermediate_size"] = hparams["mlp_hidden_size"]

        super().set_gguf_parameters()
        self._try_set_pooling_type()

        # Add parameters similar to LlamaModel
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if (rope_dim := hparams.get("head_dim")) is None:
            n_heads = hparams.get("num_attention_heads", hparams.get("n_heads"))
            assert n_heads is not None
            rope_dim = hparams["hidden_size"] // n_heads
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        # LLaDA models use non-causal attention for diffusion, similar to Dream
        self.gguf_writer.add_causal_attention(False)

        # LLaDA models don't shift their logits
        self.gguf_writer.add_diffusion_shift_logits(False)

        mask_token_id = hparams.get("mask_token_id")
        if mask_token_id is None and self.hf_arch == "LLaDAOGuiForDiffusionLM":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
            mask_token_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if mask_token_id is not None:
            self.gguf_writer.add_mask_token_id(int(mask_token_id))

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams.get("num_attention_heads", self.hparams.get("n_heads"))
        assert n_head is not None
        n_kv_head = self.hparams.get("num_key_value_heads", self.hparams.get("n_kv_heads"))

        if self.undo_permute:
            if name.endswith(("q_proj.weight", "q_proj.bias")):
                data_torch = LLaDAModel.permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight", "k_proj.bias")):
                data_torch = LLaDAModel.permute(data_torch, n_head, n_kv_head)
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                data_torch = LLaDAModel.permute(data_torch, 1, 1)

        # LLaDA model tensors should be mapped directly since it's the base model
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("LLaDAMoEModel", "LLaDAMoEModelLM")
class LLaDAMoEModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLADA_MOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (expert_intermediate_size := self.hparams.get("expert_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(expert_intermediate_size)

        self.gguf_writer.add_mask_token_id(156895)
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_diffusion_shift_logits(False)

    _experts: list[dict[str, Tensor]] | None = None

    # Copied from: Qwen2MoeModel
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    # Copied from: Qwen2MoeModel
    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
