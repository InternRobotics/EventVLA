import math
import os
from collections import namedtuple
from typing import List, Optional

import numpy as np
import torch

from eventvla.model.framework.Pi05_base import Pi05 as Pi05Base, make_att_2d_masks
from eventvla.model.framework.pi05_keyframe_mixin import Pi05KeyframeMixin
from eventvla.model.modules.vlm.pi05_mem_video_encoder import Pi05MemVisionTower
from eventvla.model.framework import FRAMEWORK_REGISTRY

ANCHOR_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


def install_mem_vision_tower(paligemma_module: torch.nn.Module, mem_vision_tower: torch.nn.Module) -> None:
    """Replace only the canonical nested vision tower path used by checkpoint keys."""
    paligemma_module.model.vision_tower = mem_vision_tower

    top_level_module = paligemma_module._modules.get("vision_tower")
    if top_level_module is mem_vision_tower:
        del paligemma_module._modules["vision_tower"]
    if paligemma_module.__dict__.get("vision_tower") is mem_vision_tower:
        del paligemma_module.__dict__["vision_tower"]


@FRAMEWORK_REGISTRY.register("Pi05MEM")
class Pi05MEM(Pi05KeyframeMixin, Pi05Base):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._init_keyframe_memory_config(config)

        self.temporal_num_frames = int(getattr(config.framework, "temporal_num_frames", 4))
        self.temporal_stride = int(getattr(config.framework, "temporal_stride", 10))
        self.temporal_anchor_first = bool(getattr(config.framework, "temporal_anchor_first", False))
        self.temporal_state_tokens = bool(getattr(config.framework, "temporal_state_tokens", False))
        self.temporal_attn_every_n_layers = int(getattr(config.framework, "temporal_attn_every_n_layers", 4))
        self.anchor_image_keys = list(ANCHOR_IMAGE_KEYS)
        self._cached_prefix_state = None

        if self.temporal_state_tokens:
            self.memory_state_proj = torch.nn.Linear(
                config.framework.action_dim,
                config.framework.paligemma_config.width,
                dtype=self.proj_dtype,
            )

        mem_vision_tower = Pi05MemVisionTower(
            self.paligemma_with_expert.paligemma.model.vision_tower,
            max_frames=max(self.temporal_num_frames, 1),
            temporal_attn_every_n_layers=self.temporal_attn_every_n_layers,
        )
        install_mem_vision_tower(self.paligemma_with_expert.paligemma, mem_vision_tower)
        if os.environ.get("EVENTVLA_COMPILE_ON_LOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
            self.paligemma_with_expert.embed_image = torch.compile(
                self.paligemma_with_expert.embed_image, mode="default"
            )
            self.denoise_step = torch.compile(self.denoise_step, mode="default")

    def _prepend_memory_role_text(
        self,
        instructions: List[str],
        raw_examples: Optional[List[dict]],
    ) -> List[str]:
        if not self.use_image_role_text or raw_examples is None:
            return list(instructions)
        updated = []
        for idx, instruction in enumerate(instructions):
            example = raw_examples[idx] if idx < len(raw_examples) else {}
            has_memory = bool(example.get("memory_keyframe_images", []))
            if has_memory:
                updated.append("Temporal observation clips and past keyframe images are provided. " + instruction)
            else:
                updated.append(instruction)
        return updated

    def _ordered_memory_keys(self, memory_clip_count: int) -> List[str]:
        keys = []
        for clip_idx in range(memory_clip_count):
            for view_key in self.anchor_image_keys:
                keys.append(f"memory_keyframe_{clip_idx:02d}_{view_key}")
        return keys

    def _ordered_image_keys(self, memory_clip_count: int) -> List[str]:
        memory_keys = self._ordered_memory_keys(memory_clip_count)
        if memory_keys and self.keyframe_image_position == "before_anchor_images":
            return memory_keys + list(self.anchor_image_keys)
        return list(self.anchor_image_keys) + memory_keys

    def _preprocess_observation(self, observation, *, train=True):
        self._cached_prefix_state = observation.state
        ordered_image_keys = list(getattr(observation, "ordered_image_keys", None) or self.anchor_image_keys)
        missing = set(ordered_image_keys).difference(observation.images)
        if missing:
            raise ValueError(f"images dict missing keys: expected {ordered_image_keys}, missing {sorted(missing)}")

        out_images = {}
        out_masks = {}
        for key in ordered_image_keys:
            image = observation.images[key]
            if image.ndim not in (4, 5):
                raise ValueError(f"Expected image {key} to have 4 or 5 dims, got {tuple(image.shape)}")
            out_images[key] = image

            if key in observation.image_masks:
                out_masks[key] = observation.image_masks[key]
            else:
                ref = observation.state if observation.state is not None else image
                out_masks[key] = torch.ones(image.shape[0], dtype=torch.bool, device=ref.device)

        return (
            [out_images[key] for key in ordered_image_keys],
            [out_masks[key] for key in ordered_image_keys],
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.tokenized_fast_action_mask,
            observation.state,
        )

    def build_gemma_inputs(
        self,
        batch_images,
        batch_instructions,
        batch_state=None,
        batch_action=None,
        image_normlize=True,
        device_=None,
    ):
        if device_ is None:
            device_ = self.device

        if self.config.framework.discrete_state_input:
            if batch_state is None:
                raise ValueError("State is required.")
        else:
            batch_state = None

        batch_language_tokens = []
        batch_tokenized_prompt_mask = []
        batch_tokenized_fast_action_mask = []
        for i in range(len(batch_instructions)):
            prompt = batch_instructions[i]
            if batch_state is None:
                tokenizer_state = None
            else:
                tokenizer_state = batch_state[i]
                if isinstance(tokenizer_state, torch.Tensor):
                    if tokenizer_state.ndim == 2:
                        tokenizer_state = tokenizer_state[-1]
                    elif tokenizer_state.ndim > 2:
                        raise ValueError(f"Unsupported tokenizer state shape: {tuple(tokenizer_state.shape)}")
                else:
                    tokenizer_state = np.asarray(tokenizer_state)
                    if tokenizer_state.ndim == 2:
                        tokenizer_state = tokenizer_state[-1]
                    elif tokenizer_state.ndim > 2:
                        raise ValueError(f"Unsupported tokenizer state shape: {tokenizer_state.shape}")

            if tokenizer_state is None:
                if batch_action is None:
                    tokens, token_masks, action_masks = self.tokenizer.tokenize(prompt, None, None)
                else:
                    tokens, token_masks, action_masks = self.tokenizer.tokenize(prompt, None, batch_action[i])
            else:
                if batch_action is None:
                    tokens, token_masks, action_masks = self.tokenizer.tokenize(prompt, tokenizer_state, None)
                else:
                    tokens, token_masks, action_masks = self.tokenizer.tokenize(prompt, tokenizer_state, batch_action[i])

            batch_language_tokens.append(tokens)
            batch_tokenized_prompt_mask.append(token_masks)
            batch_tokenized_fast_action_mask.append(action_masks)

        batch_language_tokens = torch.tensor(np.array(batch_language_tokens), device=device_)
        batch_tokenized_prompt_mask = torch.tensor(np.array(batch_tokenized_prompt_mask), device=device_)
        batch_tokenized_fast_action_mask = torch.tensor(
            np.array(batch_tokenized_fast_action_mask),
            device=device_,
        )
        return batch_images, batch_language_tokens, batch_tokenized_prompt_mask, batch_tokenized_fast_action_mask

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        cached_state = self._cached_prefix_state
        if self.temporal_state_tokens and cached_state is not None:
            if cached_state.ndim == 2:
                cached_state = cached_state[:, None, :]
            elif cached_state.ndim != 3:
                raise ValueError(f"Expected temporal state shape [B, T, D], got {tuple(cached_state.shape)}")

            if self.memory_state_proj.weight.dtype == torch.float32:
                cached_state = cached_state.to(torch.float32)

            def state_proj_func(state_tokens):
                return self.memory_state_proj(state_tokens)

            state_emb = self._apply_checkpoint(state_proj_func, cached_state)
            bsize, num_state_tokens = state_emb.shape[:2]
            embs.append(state_emb)
            pad_masks.append(torch.ones(bsize, num_state_tokens, dtype=torch.bool, device=state_emb.device))
            att_masks += [0] * num_state_tokens

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def _prepare_eval_batch_images(self, batch_images, images_mask, state, device_):
        imgs = np.asarray(batch_images, dtype=np.float32)
        if imgs.ndim == 5:
            batch_images = torch.from_numpy((imgs / 127.5) - 1.0).permute(0, 1, 4, 2, 3).contiguous()
        elif imgs.ndim == 6:
            batch_images = torch.from_numpy((imgs / 127.5) - 1.0).permute(0, 1, 2, 5, 3, 4).contiguous()
        else:
            raise ValueError(f"Unsupported eval image shape: {imgs.shape}")

        images_mask = torch.as_tensor(np.asarray(images_mask), dtype=torch.bool, device=device_)
        if state is None:
            state_tensor = None
        else:
            state_tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32, device=device_)
        return batch_images.to(device_), images_mask, state_tensor

    def _format_memory_view_like(self, image, reference: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.to(device=reference.device, dtype=reference.dtype)
        else:
            arr = np.asarray(image)
            tensor = torch.as_tensor(arr, device=reference.device, dtype=reference.dtype)

        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[-1] == 3 and tensor.shape[1] != 3:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        elif tensor.ndim == 3 and tensor.shape[-1] == 3 and tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1).contiguous()

        if tensor.is_floating_point() and tensor.numel() > 0 and float(tensor.max()) > 2.0:
            tensor = tensor / 255.0 * 2.0 - 1.0

        target_shape = tuple(reference.shape)
        if len(target_shape) == 4:
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                return torch.zeros_like(reference)
            if tensor.shape[1:] != target_shape[1:]:
                return torch.zeros_like(reference)
            if tensor.shape[0] > target_shape[0]:
                tensor = tensor[-target_shape[0]:]
            elif tensor.shape[0] < target_shape[0]:
                pad = tensor[:1].expand(target_shape[0] - tensor.shape[0], *tensor.shape[1:])
                tensor = torch.cat([pad, tensor], dim=0)
        else:
            if tensor.ndim == 4:
                tensor = tensor[-1]
            if tensor.ndim != 3 or tuple(tensor.shape) != target_shape:
                return torch.zeros_like(reference)
        return tensor.contiguous()

    def _normalize_memory_clip_views(
        self,
        memory_item,
        reference_views: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        single_frame_multiview = False
        if isinstance(memory_item, torch.Tensor):
            if memory_item.ndim == 5:
                raw_views = [memory_item[idx] for idx in range(memory_item.shape[0])]
            elif (
                memory_item.ndim == 4
                and memory_item.shape[0] == len(reference_views)
                and (memory_item.shape[1] in {1, 3} or memory_item.shape[-1] in {1, 3})
            ):
                raw_views = [memory_item[idx] for idx in range(memory_item.shape[0])]
                single_frame_multiview = True
            elif memory_item.ndim == 4:
                raw_views = [memory_item]
            else:
                raw_views = []
        elif isinstance(memory_item, np.ndarray):
            if memory_item.ndim == 5:
                raw_views = [memory_item[idx] for idx in range(memory_item.shape[0])]
            elif (
                memory_item.ndim == 4
                and memory_item.shape[0] == len(reference_views)
                and (memory_item.shape[1] in {1, 3} or memory_item.shape[-1] in {1, 3})
            ):
                raw_views = [memory_item[idx] for idx in range(memory_item.shape[0])]
                single_frame_multiview = True
            elif memory_item.ndim == 4:
                raw_views = [memory_item]
            else:
                raw_views = []
        elif isinstance(memory_item, (list, tuple)):
            raw_views = list(memory_item)
            if len(raw_views) == len(reference_views):
                raw_ndims = []
                for view in raw_views:
                    ndim = getattr(view, "ndim", None)
                    if ndim is None:
                        ndim = np.asarray(view).ndim
                    raw_ndims.append(ndim)
                single_frame_multiview = all(ndim == 3 for ndim in raw_ndims)
        else:
            raw_views = []

        formatted_views = []
        for view_idx, reference in enumerate(reference_views):
            if single_frame_multiview and reference.ndim == 4:
                reference = reference[-1]
            if view_idx < len(raw_views):
                formatted_views.append(self._format_memory_view_like(raw_views[view_idx], reference))
            else:
                formatted_views.append(torch.zeros_like(reference))
        return formatted_views

    @staticmethod
    def _black_image_like(reference: torch.Tensor) -> torch.Tensor:
        if reference.is_floating_point():
            return torch.full_like(reference, -1.0)
        return torch.zeros_like(reference)

    def _append_keyframe_memory_to_anchor_temporal(
        self,
        batch_images: torch.Tensor,
        raw_examples: List[dict],
        max_memory_count: int,
    ) -> torch.Tensor:
        if batch_images.ndim != 6:
            raise ValueError(f"Expected anchor images [B, V, T, C, H, W], got {tuple(batch_images.shape)}")

        batch_size, image_num = batch_images.shape[:2]
        appended_views = []
        for sample_idx in range(batch_size):
            example = raw_examples[sample_idx] if sample_idx < len(raw_examples) else {}
            raw_memory = list(example.get("memory_keyframe_images", []) or [])[-max_memory_count:]
            reference_views = [
                batch_images[sample_idx, view_idx, -1]
                for view_idx in range(image_num)
            ]
            formatted_memory = [
                self._normalize_memory_clip_views(memory_item, reference_views)
                for memory_item in raw_memory
            ]
            missing_count = max_memory_count - len(formatted_memory)
            sample_views = []
            for view_idx in range(image_num):
                pad_frame = self._black_image_like(batch_images[sample_idx, view_idx, -1])
                memory_frames = [pad_frame for _ in range(max(0, missing_count))]
                memory_frames.extend(memory[view_idx] for memory in formatted_memory)
                current_frames = [batch_images[sample_idx, view_idx, frame_idx] for frame_idx in range(batch_images.shape[2])]
                sample_views.append(torch.stack(memory_frames + current_frames, dim=0))
            appended_views.append(torch.stack(sample_views, dim=0))
        return torch.stack(appended_views, dim=0).contiguous()

    def _build_observation(
        self,
        batch_images,
        images_mask,
        batch_tokenized_prompt,
        batch_tokenized_prompt_mask,
        batch_tokenized_fast_action_mask,
        state,
        raw_examples: Optional[List[dict]] = None,
    ):
        Observation = namedtuple(
            "Observation",
            [
                "images",
                "image_masks",
                "tokenized_prompt",
                "tokenized_prompt_mask",
                "tokenized_fast_action_mask",
                "state",
                "token_ar_mask",
                "token_loss_mask",
                "ordered_image_keys",
            ],
        )

        batch_size, image_num = batch_images.shape[:2]
        if image_num != len(self.anchor_image_keys):
            raise ValueError(f"Expected {len(self.anchor_image_keys)} anchor views, got {image_num}")

        append_memory_to_anchor_temporal = (
            self.use_keyframe_image_memory
            and raw_examples is not None
            and self.memory_injection_mode == "pimem_keyframe_single_frame_3view"
        )
        append_memory_count = 0
        if append_memory_to_anchor_temporal:
            append_memory_count = max(0, int(self.max_keyframe_images))
            if append_memory_count > 0:
                batch_images = self._append_keyframe_memory_to_anchor_temporal(
                    batch_images=batch_images,
                    raw_examples=raw_examples,
                    max_memory_count=append_memory_count,
                )

        batch_images_dict = {
            key: batch_images[:, idx]
            for idx, key in enumerate(self.anchor_image_keys)
        }
        batch_images_mask_dict = {
            key: images_mask[:, idx]
            for idx, key in enumerate(self.anchor_image_keys)
        }

        max_memory_count = 0
        if self.use_keyframe_image_memory and raw_examples is not None and not append_memory_to_anchor_temporal:
            max_memory_count = max(
                (len(list(example.get("memory_keyframe_images", []) or [])) for example in raw_examples),
                default=0,
            )
            max_memory_count = min(max_memory_count, max(0, int(self.max_keyframe_images)))

        if max_memory_count > 0:
            anchor_reference_views = [
                [batch_images[sample_idx, view_idx] for view_idx in range(image_num)]
                for sample_idx in range(batch_size)
            ]
            formatted_memory_by_sample = []
            for sample_idx in range(batch_size):
                example = raw_examples[sample_idx] if raw_examples is not None and sample_idx < len(raw_examples) else {}
                raw_memory = list(example.get("memory_keyframe_images", []) or [])[-max_memory_count:]
                formatted_clips = [
                    self._normalize_memory_clip_views(memory_item, anchor_reference_views[sample_idx])
                    for memory_item in raw_memory
                ]
                formatted_memory_by_sample.append(formatted_clips)

            for memory_idx in range(max_memory_count):
                for view_idx, view_key in enumerate(self.anchor_image_keys):
                    key = f"memory_keyframe_{memory_idx:02d}_{view_key}"
                    per_sample_views = []
                    per_sample_mask = []
                    for sample_idx in range(batch_size):
                        if memory_idx < len(formatted_memory_by_sample[sample_idx]):
                            per_sample_views.append(formatted_memory_by_sample[sample_idx][memory_idx][view_idx])
                            per_sample_mask.append(True)
                        else:
                            per_sample_views.append(None)
                            per_sample_mask.append(False)
                    reference_view = next((view for view in per_sample_views if view is not None), None)
                    if reference_view is None:
                        reference_view = anchor_reference_views[0][view_idx]
                    per_sample_views = [
                        view if view is not None else torch.zeros_like(reference_view)
                        for view in per_sample_views
                    ]
                    batch_images_dict[key] = torch.stack(per_sample_views, dim=0)
                    batch_images_mask_dict[key] = torch.tensor(
                        per_sample_mask,
                        device=images_mask.device,
                        dtype=torch.bool,
                    )

        ordered_image_keys = self._ordered_image_keys(max_memory_count)
        return Observation(
            images=batch_images_dict,
            image_masks=batch_images_mask_dict,
            tokenized_prompt=batch_tokenized_prompt,
            tokenized_prompt_mask=batch_tokenized_prompt_mask,
            tokenized_fast_action_mask=batch_tokenized_fast_action_mask,
            state=state,
            token_ar_mask=None,
            token_loss_mask=None,
            ordered_image_keys=ordered_image_keys,
        )

    def _build_model_observation(
        self,
        batch_images,
        images_mask,
        instructions: List[str],
        state,
        raw_examples: Optional[List[dict]] = None,
        batch_action=None,
        device_=None,
    ):
        updated_instructions = self._prepend_memory_role_text(list(instructions), raw_examples)
        (
            batch_images,
            batch_tokenized_prompt,
            batch_tokenized_prompt_mask,
            batch_tokenized_fast_action_mask,
        ) = self.build_gemma_inputs(
            batch_images,
            updated_instructions,
            state,
            batch_action=batch_action,
            device_=device_,
        )
        if state is not None:
            state = self.pad_state_action(state).squeeze(1)
        return self._build_observation(
            batch_images,
            images_mask,
            batch_tokenized_prompt,
            batch_tokenized_prompt_mask,
            batch_tokenized_fast_action_mask,
            state,
            raw_examples=raw_examples,
        )

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: torch.Tensor,
        images_mask: torch.Tensor,
        instructions: List[str],
        state: torch.Tensor = None,
        num_steps: int = 10,
        **kwargs: str,
    ) -> np.ndarray:
        device_ = self.action_in_proj.weight.device
        num_steps = int(kwargs.get("num_ddim_steps", num_steps))
        raw_examples = kwargs.get("raw_examples", None)
        if raw_examples is None and isinstance(kwargs.get("examples", None), dict):
            raw_examples = self._extract_raw_examples(kwargs["examples"])
        elif raw_examples is not None:
            raw_examples = list(raw_examples)

        if kwargs.get("run_eval", False):
            batch_images, images_mask, state = self._prepare_eval_batch_images(batch_images, images_mask, state, device_)

        bsize, image_num = batch_images.shape[0], batch_images.shape[1]
        if image_num != len(self.anchor_image_keys):
            raise ValueError(f"Expected {len(self.anchor_image_keys)} views, got {image_num}")
        if images_mask.shape[1] != len(self.anchor_image_keys):
            raise ValueError(f"Expected image mask width {len(self.anchor_image_keys)}, got {tuple(images_mask.shape)}")

        batch_observations = self._build_model_observation(
            batch_images=batch_images,
            images_mask=images_mask,
            instructions=list(instructions),
            state=state,
            raw_examples=raw_examples,
            device_=device_,
        )

        actions_shape = (bsize, self.config.framework.action_horizon, self.config.framework.action_dim)
        noise = self.sample_noise(actions_shape, device_)
        images, img_masks, lang_tokens, lang_masks, _, state = self._preprocess_observation(batch_observations, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device_)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device_)
        last_suffix_out = None
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t, last_suffix_out = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                return_hidden=True,
            )
            x_t = x_t + dt * v_t
            time += dt

        output = {"normalized_actions": x_t.cpu().numpy()}
        if last_suffix_out is not None:
            _, keyframe_outputs = self._compute_keyframe_outputs(
                suffix_out=last_suffix_out,
                raw_examples=raw_examples,
                compute_loss=False,
            )
            output.update(self._prediction_keyframe_outputs(keyframe_outputs))
        return output

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        return_hidden: bool = False,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.framework.action_horizon :].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if return_hidden:
            return v_t, suffix_out
        return v_t
