from typing import List, Optional

import torch
from torch import nn


KEYFRAME_IMAGE_MEMORY_MODES = frozenset(
    {"pure_image_keyframe_memory", "pimem_keyframe_clip_memory", "pimem_keyframe_single_frame_3view"}
)


class Pi05KeyframeMixin:
    def _init_keyframe_memory_config(self, config) -> None:
        framework_cfg = getattr(config, "framework", None)
        datasets_cfg = getattr(config, "datasets", None)
        vla_data_cfg = getattr(datasets_cfg, "vla_data", None)
        memory_cfg = getattr(config.framework, "memory_buffer", None)

        def cfg_value(container, name: str, default):
            if container is None:
                return default
            if hasattr(container, "get"):
                return container.get(name, default)
            return getattr(container, name, default)

        injection_cfg = cfg_value(memory_cfg, "qwen_memory_injection", {})
        self.memory_ablation_mode = str(cfg_value(framework_cfg, "memory_ablation_mode", "baseline_no_memory")).lower()
        self.memory_injection_enabled = bool(cfg_value(injection_cfg, "enabled", True))
        self.memory_injection_mode = str(cfg_value(injection_cfg, "mode", self.memory_ablation_mode)).lower()
        self.use_keyframe_image_memory = self.memory_injection_enabled and self.memory_injection_mode in KEYFRAME_IMAGE_MEMORY_MODES

        keyframe_image_cfg = cfg_value(vla_data_cfg, "keyframe_image_memory", {}) or {}
        self.max_keyframe_images = int(cfg_value(injection_cfg, "max_keyframe_images", cfg_value(keyframe_image_cfg, "max_keyframes", 4)))
        self.keyframe_image_position = str(cfg_value(injection_cfg, "keyframe_image_position", "after_anchor_images_before_action")).lower()
        self.use_image_role_text = bool(cfg_value(injection_cfg, "use_image_role_text", True))
        self.keyframe_threshold = float(cfg_value(memory_cfg, "keyframe_threshold", 0.5))
        self.event_future_min_offset = max(0, int(cfg_value(memory_cfg, "event_future_min_offset", 1)))
        self.event_commit_threshold = float(cfg_value(memory_cfg, "event_commit_threshold", 0.55))

        raw_head_mode = cfg_value(memory_cfg, "use_keyframe_predict_head", "enabled")
        if isinstance(raw_head_mode, bool):
            self.keyframe_predict_head_mode = "enabled" if raw_head_mode else "disabled"
        else:
            self.keyframe_predict_head_mode = str(raw_head_mode).lower()
        if self.keyframe_predict_head_mode in {"true", "yes", "on", "1"}:
            self.keyframe_predict_head_mode = "enabled"
        elif self.keyframe_predict_head_mode in {"false", "no", "off", "0", "none"}:
            self.keyframe_predict_head_mode = "disabled"
        elif self.keyframe_predict_head_mode not in {"enabled", "disabled", "auto"}:
            self.keyframe_predict_head_mode = "enabled"

        self.chunk_len = int(config.framework.action_horizon)
        self.keyframe_memory_teacher_prob = 0.0
        self.keyframe_schedule_progress = 1.0
        self.keyframe_head = nn.Sequential(
            nn.LayerNorm(config.framework.action_expert_config.width),
            nn.Linear(config.framework.action_expert_config.width, config.framework.action_expert_config.width),
            nn.GELU(),
            nn.Linear(config.framework.action_expert_config.width, 1),
        )
        self.register_buffer("_keyframe_annotations_observed", torch.tensor(False, dtype=torch.bool), persistent=True)

    def _extract_raw_examples(self, examples: Optional[dict]) -> Optional[List[dict]]:
        if not isinstance(examples, dict):
            return None
        raw_examples = examples.get("raw_examples")
        return None if raw_examples is None else list(raw_examples)

    def _should_use_keyframe_predict_head(self) -> bool:
        if self.keyframe_predict_head_mode == "enabled":
            return True
        if self.keyframe_predict_head_mode == "disabled":
            return False
        return bool(self._keyframe_annotations_observed.detach().cpu().item())

    def _empty_keyframe_predictions(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        probs = torch.zeros((batch_size, self.chunk_len), device=device, dtype=dtype)
        pred_mask = torch.zeros((batch_size, self.chunk_len), device=device, dtype=torch.bool)
        event_offset = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        event_confidence = torch.zeros((batch_size,), device=device, dtype=dtype)
        should_commit = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        return probs, pred_mask, event_offset, event_confidence, should_commit

    def _select_chunk_event(self, chunk_keyframe_probs: torch.Tensor, threshold: Optional[float] = None):
        batch_size, chunk_len = chunk_keyframe_probs.shape
        start_offset = min(max(int(self.event_future_min_offset), 0), chunk_len)
        if start_offset >= chunk_len:
            return (
                torch.full((batch_size,), -1, device=chunk_keyframe_probs.device, dtype=torch.long),
                torch.zeros((batch_size,), device=chunk_keyframe_probs.device, dtype=chunk_keyframe_probs.dtype),
                torch.zeros((batch_size,), device=chunk_keyframe_probs.device, dtype=torch.bool),
            )
        future_probs = chunk_keyframe_probs[:, start_offset:]
        confidence, rel_offset = future_probs.max(dim=1)
        event_offset = rel_offset + start_offset
        should_commit = confidence >= (self.event_commit_threshold if threshold is None else float(threshold))
        return event_offset.long(), confidence, should_commit

    def _compute_keyframe_outputs(self, suffix_out: torch.Tensor, raw_examples=None, compute_loss: bool = False):
        if self._should_use_keyframe_predict_head():
            logits = self.keyframe_head(suffix_out).squeeze(-1)
            probs = torch.sigmoid(logits)
            pred_mask = probs >= self.keyframe_threshold
            event_offset, event_confidence, should_commit = self._select_chunk_event(probs)
        else:
            probs, pred_mask, event_offset, event_confidence, should_commit = self._empty_keyframe_predictions(
                suffix_out.shape[0], suffix_out.device, suffix_out.dtype
            )

        zero = suffix_out.new_zeros(())
        output = {
            "chunk_keyframe_prob": probs.detach(),
            "chunk_keyframe_pred_mask": pred_mask.detach(),
            "pred_event_offset": event_offset.detach(),
            "pred_event_confidence": event_confidence.detach(),
            "should_trigger_event": should_commit.detach(),
            "keyframe_prob": event_confidence.detach(),
            "predicted_is_keyframe": should_commit.detach(),
            "memory_is_keyframe": should_commit.detach(),
            "keyframe_memory_rate": should_commit.float().mean().detach(),
            "keyframe_head_enabled": torch.tensor(float(self._should_use_keyframe_predict_head()), device=suffix_out.device),
            "keyframe_annotation_rate": torch.tensor(1.0, device=suffix_out.device),
            "keyframe_memory_teacher_prob": torch.tensor(float(self.keyframe_memory_teacher_prob), device=suffix_out.device),
            "keyframe_memory_teacher_usage": torch.tensor(0.0, device=suffix_out.device),
            "keyframe_memory_predict_usage": torch.tensor(1.0, device=suffix_out.device),
            "keyframe_memory_schedule_progress": torch.tensor(float(self.keyframe_schedule_progress), device=suffix_out.device),
        }
        return zero, output

    @staticmethod
    def _to_prediction_value(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.item() if value.ndim == 0 else value.numpy()
        return value

    def _prediction_keyframe_outputs(self, keyframe_outputs: dict) -> dict:
        prediction_keys = (
            "chunk_keyframe_prob",
            "chunk_keyframe_pred_mask",
            "pred_event_offset",
            "pred_event_confidence",
            "should_trigger_event",
            "keyframe_prob",
            "predicted_is_keyframe",
            "memory_is_keyframe",
            "keyframe_memory_rate",
            "keyframe_head_enabled",
            "keyframe_annotation_rate",
            "keyframe_memory_teacher_prob",
            "keyframe_memory_teacher_usage",
            "keyframe_memory_predict_usage",
            "keyframe_memory_schedule_progress",
        )
        return {key: self._to_prediction_value(value) for key, value in keyframe_outputs.items() if key in prediction_keys}
