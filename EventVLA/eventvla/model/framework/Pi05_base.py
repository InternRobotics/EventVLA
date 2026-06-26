import logging
import math
import os
import pathlib
from copy import deepcopy
from typing import Optional

import numpy as np
import sentencepiece
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from eventvla.model.modules.vlm.PaliGemma import PaliGemmaWithExpertModel
from eventvla.model.framework.base_framework import baseframework


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48, tokenizer_path: str | None = None):
        self._max_len = max_len
        path = self._resolve_tokenizer_path(tokenizer_path)
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    @staticmethod
    def _resolve_tokenizer_path(tokenizer_path: str | None = None) -> pathlib.Path:
        deploy_root = pathlib.Path(__file__).resolve().parents[3]
        paligemma_model_path = os.environ.get("PALIGEMMA_MODEL_PATH")
        paligemma_tokenizer = (
            pathlib.Path(paligemma_model_path) / "tokenizer.model"
            if paligemma_model_path
            else None
        )
        candidates = (
            tokenizer_path,
            os.environ.get("PALIGEMMA_TOKENIZER_PATH"),
            paligemma_tokenizer,
            deploy_root / "playground/tokenizer/paligemma_tokenizer.model",
            pathlib.Path.cwd() / "playground/tokenizer/paligemma_tokenizer.model",
        )
        for candidate in candidates:
            if not candidate:
                continue
            path = pathlib.Path(candidate)
            if path.exists():
                return path
        raise FileNotFoundError(
            "Missing paligemma_tokenizer.model. Set PALIGEMMA_TOKENIZER_PATH or keep "
            "playground/tokenizer/paligemma_tokenizer.model inside eventvla_deploy."
        )

    def tokenize(self, prompt: str, state=None, action: list | None = None):
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            state_value = deepcopy(state)
            if hasattr(state_value, "detach"):
                state_np = state_value.detach().cpu().numpy()
            else:
                state_np = np.asarray(state_value)

            pad_num = -2
            pad_index = np.where(np.abs(state_np - pad_num) < 1e-5)
            state_np = np.clip(state_np, -1, 1)
            state_np[pad_index] = pad_num
            if len(state_np.shape) != 1:
                raise ValueError(f"state dim must be (D,), got {state_np.shape}")

            discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        action_mask = []
        if action is not None:
            action_mask = [True] * len(tokens) + [False] * len(action)
            tokens = tokens + action

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            if action is not None:
                action_mask = action_mask + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    "Token length (%d) exceeds max length (%d), truncating.",
                    len(tokens),
                    self._max_len,
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
            if action is not None:
                action_mask = action_mask[: self._max_len]

        if len(action_mask) == 0:
            action_mask = mask

        return np.asarray(tokens), np.asarray(mask), np.asarray(action_mask)


class Pi05(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config
        self.pi05 = config.framework.pi05
        
        self.proj_dtype = torch.float32

        paligemma_config = config.framework.paligemma_config 
        action_expert_config = config.framework.action_expert_config 
        
        self.tokenizer = PaligemmaTokenizer(config.framework.max_token_len)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision='bfloat16' if config.framework.precision == 'bfloat16' else 'float32',
            pretrained_model_path=getattr(config.framework, "paligemma_model_path", None),
        )

        self.action_in_proj = nn.Linear(config.framework.action_dim, action_expert_config.width, dtype=self.proj_dtype)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.framework.action_dim, dtype=self.proj_dtype)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width, dtype=self.proj_dtype)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width, dtype=self.proj_dtype)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width, dtype=self.proj_dtype)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width, dtype=self.proj_dtype)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width, dtype=self.proj_dtype)

        torch.set_float32_matmul_precision("high")
        # self.predict_action = torch.compile(self.predict_action, mode="max-autotune")
        self._torch_compile_enabled_targets = set()

        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def enable_torch_compile(
        self,
        targets="vision,denoise",
        *,
        mode: str | None = "default",
        backend: str | None = None,
        fullgraph: bool = False,
        dynamic: bool | None = None,
        suppress_errors: bool = True,
    ) -> list[str]:
        """Compile stable deployment hot paths without compiling the whole policy.

        The full `predict_action` path contains Python-side tokenization, memory
        packing, and output conversion. Compile only the tensor-heavy pieces so
        deployment can enable this explicitly without changing training behavior.
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")

        if isinstance(targets, str):
            requested = {item.strip().lower() for item in targets.split(",") if item.strip()}
        else:
            requested = {str(item).strip().lower() for item in targets if str(item).strip()}
        if "all" in requested:
            requested.update({"vision", "denoise"})

        compile_kwargs = {"fullgraph": bool(fullgraph)}
        if mode:
            compile_kwargs["mode"] = mode
        if backend:
            compile_kwargs["backend"] = backend
        if dynamic is not None:
            compile_kwargs["dynamic"] = bool(dynamic)

        try:
            import torch._dynamo as _dynamo

            _dynamo.config.suppress_errors = bool(suppress_errors)
        except Exception as exc:
            logging.warning("Could not set torch._dynamo.config.suppress_errors: %s", exc)

        compiled = []
        if "vision" in requested and "vision" not in self._torch_compile_enabled_targets:
            self.paligemma_with_expert.embed_image = torch.compile(
                self.paligemma_with_expert.embed_image,
                **compile_kwargs,
            )
            self._torch_compile_enabled_targets.add("vision")
            compiled.append("vision")

        if "denoise" in requested and "denoise" not in self._torch_compile_enabled_targets:
            self.denoise_step = torch.compile(self.denoise_step, **compile_kwargs)
            self._torch_compile_enabled_targets.add("denoise")
            compiled.append("denoise")

        unknown = requested.difference({"all", "vision", "denoise"})
        if unknown:
            logging.warning("Ignoring unknown torch.compile target(s): %s", sorted(unknown))
        return compiled

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def embed_suffix(self, state, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.framework.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def pad_state_action(self, data, dtype='float32'):
        dtype_ = torch.float32 if dtype=="float32" else torch.bfloat16
        return data.to(dtype=dtype_)
