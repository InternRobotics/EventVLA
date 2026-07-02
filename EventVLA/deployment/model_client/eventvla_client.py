#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client
from PIL import Image as PILImage

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CONTROL_HZ = 30.0
ACTION_DIM = 14
ACTION_CHUNK_SIZE = 50
MODEL_STATE_DIM = 32
STATE_PAD_VALUE = -2.0
NORM_TYPE = "mean_std"
STATE_NORM_TYPE = "mean_std"

DEFAULT_INSTRUCTION = ("Pick up and put down the bottles on the table in the order which they are pointed by the stick one by one at the beginning.")

IMG_HEAD_TOPIC = "/camera/camera_h/color/image_rect_raw/compressed"
IMG_LEFT_TOPIC = "/camera/camera_l/color/image_rect_raw/compressed"
IMG_RIGHT_TOPIC = "/camera/camera_r/color/image_rect_raw/compressed"
CMD_LEFT_TOPIC = "/arm_master_l_cmd"
CMD_RIGHT_TOPIC = "/arm_master_r_cmd"
MON_LEFT_TOPIC = "/arm_master_l_status"
MON_RIGHT_TOPIC = "/arm_master_r_status"
FB_LEFT_TOPIC = "/arm_slave_l_status"
FB_RIGHT_TOPIC = "/arm_slave_r_status"
JOY_TOPIC = "/arx_joy"

JOY_START_BUTTON = 0
JOY_PAUSE_BUTTON = 1
JOY_NEXT_INSTRUCTION_BUTTON = 2
JOY_RESET_BUTTON = 3

RESET_HOLD_SEC = 1.0
RESET_WAIT_STATE_TIMEOUT_SEC = 5.0
RESET_INTERP_MAX_STEP_RAD = 0.015
RESET_INTERP_HZ = 60.0
GRIPPER_OPEN_VALUE = -4.0
RESET_ACTION = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN_VALUE,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN_VALUE],
    dtype=np.float32,
)


@dataclass
class RuntimeConfig:
    ws_host: str
    ws_port: int
    stats_json: str
    unnorm_key: str
    instructions: list[str]
    temporal_frames: int
    temporal_stride: int
    temporal_anchor_first: bool
    keyframes: int
    keyframe_threshold: float
    keyframe_min_gap: int
    keyframe_cooldown: int
    keyframe_merge_window: int
    gripper_boost: float
    gripper_boost_max: float
    gripper_close_threshold: float
    debug_interval: int
    reset_on_start: bool


def _pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for msgpack serialization: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _pick_norm_keys(norm_type: str) -> tuple[str, str]:
    if norm_type == "mean_std":
        return "mean", "std"
    if norm_type == "min_max":
        return "min", "max"
    if norm_type == "q01_q99":
        return "q01", "q99"
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _normalize_values(x: np.ndarray, stats_a: np.ndarray, stats_b: np.ndarray, norm_type: str) -> np.ndarray:
    if norm_type in {"min_max", "q01_q99"}:
        return 2.0 * (x - stats_a) / (stats_b - stats_a + 1e-8) - 1.0
    if norm_type == "mean_std":
        return (x - stats_a) / (stats_b + 1e-8)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _unnormalize_values(x: np.ndarray, stats_a: np.ndarray, stats_b: np.ndarray, norm_type: str) -> np.ndarray:
    if norm_type in {"min_max", "q01_q99"}:
        return 0.5 * (x + 1.0) * (stats_b - stats_a) + stats_a
    if norm_type == "mean_std":
        return x * stats_b + stats_a
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _to_1d_float32(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.ascontiguousarray(arr)


def _coerce_normalized_actions(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return None if value.size == 0 else value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return None if not value else _as_python_scalar(value[0])
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _safe_int(value: Any, default: int = -1) -> int:
    scalar = _as_python_scalar(value)
    try:
        return int(scalar)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    scalar = _as_python_scalar(value)
    try:
        return float(scalar)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    scalar = _as_python_scalar(value)
    if scalar is None:
        return bool(default)
    if isinstance(scalar, str):
        return scalar.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(scalar)


def _resize_with_pad(image: np.ndarray, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    pil = PILImage.fromarray(arr)
    cur_w, cur_h = pil.size
    if cur_w == width and cur_h == height:
        return np.ascontiguousarray(arr)
    ratio = max(cur_w / float(width), cur_h / float(height))
    resized_w = max(1, int(cur_w / ratio))
    resized_h = max(1, int(cur_h / ratio))
    resized = pil.resize((resized_w, resized_h), resample=PILImage.BILINEAR)
    canvas = PILImage.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - resized_w) // 2, (height - resized_h) // 2))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def _summarize_keyframe_infer_data(infer_data: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("should_trigger_event", "pred_event_offset", "pred_event_confidence", "keyframe_prob"):
        if key in infer_data:
            summary[key] = _as_python_scalar(infer_data[key])
    probs = np.asarray(infer_data.get("chunk_keyframe_prob", []), dtype=np.float32).reshape(-1)
    if probs.size:
        idx = int(np.argmax(probs))
        summary["chunk_keyframe_prob_max"] = float(probs[idx])
        summary["chunk_keyframe_prob_argmax"] = idx
    return summary


class TemporalObservationBuffer:
    """Keep the anchor/current temporal observation contract for Pi05MEM."""

    def __init__(self, *, num_frames: int, stride: int, anchor_first: bool, maxlen: int = 128) -> None:
        self.num_frames = max(1, int(num_frames))
        self.stride = max(1, int(stride))
        self.anchor_first = bool(anchor_first)
        self.maxlen = max(int(maxlen), self.num_frames + self.stride + 1)
        self.reset()

    def reset(self) -> None:
        self._history: deque[dict[str, Any]] = deque(maxlen=self.maxlen)
        self._anchor: dict[str, Any] | None = None
        self._step = -1

    def add(self, *, head_img: np.ndarray, left_img: np.ndarray, right_img: np.ndarray, state_vec: np.ndarray) -> None:
        self._step += 1
        sample = {
            "step": self._step,
            "images": (
                _resize_with_pad(head_img),
                _resize_with_pad(left_img),
                _resize_with_pad(right_img),
            ),
            "state": np.asarray(state_vec, dtype=np.float32).copy(),
        }
        self._history.append(sample)
        if self._anchor is None:
            self._anchor = sample

    def ready(self) -> bool:
        return bool(self._history and self._anchor is not None)

    def current_step(self) -> int:
        return -1 if not self._history else int(self._history[-1]["step"])

    def _sample_for_step(self, target_step: int) -> dict[str, Any]:
        for item in reversed(self._history):
            if int(item["step"]) <= int(target_step):
                return item
        return self._history[0]

    def _build_indices(self, current_step: int, anchor_step: int) -> list[int]:
        if self.num_frames == 1:
            return [current_step]
        if self.anchor_first:
            if self.num_frames == 2:
                return [anchor_step, current_step]
            indices = [anchor_step]
            for gap_idx in range(self.num_frames - 2, 0, -1):
                indices.append(max(anchor_step, current_step - gap_idx * self.stride))
            indices.append(current_step)
            return indices
        return [max(anchor_step, current_step - (self.num_frames - 1 - i) * self.stride) for i in range(self.num_frames)]

    def build_payload_tensors(self) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
        if not self.ready():
            raise RuntimeError("Temporal observation buffer is empty")
        current = self._history[-1]
        anchor = self._anchor
        assert anchor is not None
        indices = self._build_indices(current_step=int(current["step"]), anchor_step=int(anchor["step"]))
        selected = [anchor if step == int(anchor["step"]) else self._sample_for_step(step) for step in indices]
        temporal_views = [
            np.ascontiguousarray(np.stack([item["images"][cam_idx] for item in selected], axis=0), dtype=np.uint8)
            for cam_idx in range(3)
        ]
        temporal_state = np.stack([item["state"] for item in selected], axis=0).astype(np.float32, copy=False)
        return temporal_views, np.ascontiguousarray(temporal_state), indices

    def build_single_frame_views_for_step(self, target_step: int) -> tuple[list[np.ndarray], int]:
        if not self.ready():
            raise RuntimeError("Temporal observation buffer is empty")
        anchor = self._anchor
        assert anchor is not None
        step = min(max(int(target_step), int(anchor["step"])), self.current_step())
        sample = anchor if step == int(anchor["step"]) else self._sample_for_step(step)
        return [np.ascontiguousarray(sample["images"][i], dtype=np.uint8) for i in range(3)], int(sample["step"])


class PredictedKeyframeMemory:
    """Store scheduled predicted keyframes as single-frame 3-view memories."""

    def __init__(self, cfg: RuntimeConfig) -> None:
        self.enabled = cfg.keyframes > 0
        self.max_keyframes = max(0, int(cfg.keyframes))
        self.threshold = float(cfg.keyframe_threshold)
        self.min_gap = max(0, int(cfg.keyframe_min_gap))
        self.cooldown = max(0, int(cfg.keyframe_cooldown))
        self.merge_window = max(0, int(cfg.keyframe_merge_window))
        self.reset()

    def reset(self) -> None:
        self._memory: deque[dict[str, Any]] = deque(maxlen=max(0, self.max_keyframes))
        self._pending: list[dict[str, Any]] = []
        self._committed_steps: set[int] = set()
        self._last_committed_step = -1
        self._next_allowed_step = 0

    def memory_steps(self) -> list[int]:
        return [int(item["step"]) for item in self._memory]

    def pending_steps(self) -> list[int]:
        return [int(item["step"]) for item in self._pending]

    def build_raw_examples(self) -> list[dict[str, Any]] | None:
        if not self.enabled or self.max_keyframes <= 0:
            return None
        return [{
            "memory_keyframe_images": [item["frame"] for item in self._memory],
            "memory_keyframe_steps": self.memory_steps(),
            "memory_keyframe_count": len(self._memory),
        }]

    def register_prediction(
        self,
        *,
        infer_data: dict[str, Any],
        base_step: int,
        action_chunk_len: int,
    ) -> tuple[bool, int, float, str]:
        if not self.enabled:
            return False, -1, 0.0, "disabled"
        base_step = int(base_step)
        if base_step < self._next_allowed_step:
            return False, -1, 0.0, f"cooldown_until_{self._next_allowed_step}"
        confidence = _safe_float(infer_data.get("pred_event_confidence", infer_data.get("keyframe_prob", 0.0)), 0.0)
        if confidence < self.threshold:
            return False, -1, confidence, "below_threshold"
        should_commit = _safe_bool(infer_data.get("should_trigger_event", infer_data.get("predicted_is_keyframe", True)), True)
        if not should_commit:
            return False, -1, confidence, "model_rejected"
        offset = _safe_int(infer_data.get("pred_event_offset", -1), -1)
        if offset < 0:
            return False, -1, confidence, "invalid_offset"
        if action_chunk_len > 0:
            offset = min(offset, max(0, int(action_chunk_len) - 1))
        step = base_step + offset
        if step in self._committed_steps:
            return False, step, confidence, "already_committed"
        if self._last_committed_step >= 0 and step - self._last_committed_step < self.min_gap:
            return False, step, confidence, "min_gap"
        for pending in self._pending:
            if abs(int(pending["step"]) - step) <= self.merge_window:
                if confidence > float(pending["confidence"]):
                    pending.update(step=step, confidence=confidence)
                    self._pending.sort(key=lambda item: int(item["step"]))
                    self._next_allowed_step = base_step + self.cooldown
                    return True, step, confidence, "merged_pending_update"
                return False, int(pending["step"]), confidence, "merged_pending_keep"
        self._pending.append({"step": step, "confidence": confidence})
        self._pending.sort(key=lambda item: int(item["step"]))
        self._next_allowed_step = base_step + self.cooldown
        return True, step, confidence, "scheduled"

    def commit_due(self, *, current_step: int, temporal_buffer: TemporalObservationBuffer) -> list[int]:
        if not self.enabled:
            return []
        committed: list[int] = []
        while self._pending and int(self._pending[0]["step"]) <= int(current_step):
            pending = self._pending.pop(0)
            step = int(pending["step"])
            if step in self._committed_steps:
                continue
            if self._last_committed_step >= 0 and step - self._last_committed_step < self.min_gap:
                logging.info("skip keyframe step=%d: min_gap from %d", step, self._last_committed_step)
                continue
            frame_views, actual_step = temporal_buffer.build_single_frame_views_for_step(step)
            self._memory.append({
                "step": step,
                "actual_step": actual_step,
                "frame": [np.ascontiguousarray(view, dtype=np.uint8) for view in frame_views],
            })
            self._committed_steps.add(step)
            self._last_committed_step = step
            self._next_allowed_step = max(self._next_allowed_step, step + self.cooldown)
            committed.append(step)
        return committed


class StatsAdapter:
    """State normalization and action unnormalization for the real A1 layout."""

    def __init__(self, stats_json: str, unnorm_key: str) -> None:
        with open(stats_json, "r", encoding="utf-8") as f:
            stats_root = json.load(f)
        if not isinstance(stats_root, dict) or not stats_root:
            raise ValueError(f"Invalid stats json: {stats_json}")
        key = unnorm_key or next(iter(stats_root.keys()))
        if key not in stats_root:
            raise KeyError(f"unnorm_key `{key}` not found in stats json keys: {list(stats_root.keys())}")
        item = stats_root[key]
        action_stats = item.get("action")
        state_stats = item.get("state")
        if not isinstance(action_stats, dict) or not isinstance(state_stats, dict):
            raise ValueError(f"Missing `{key}.action` or `{key}.state` in stats json")
        a0, a1 = _pick_norm_keys(NORM_TYPE)
        s0, s1 = _pick_norm_keys(STATE_NORM_TYPE)
        self.action_low = np.asarray(action_stats[a0], dtype=np.float32)
        self.action_high = np.asarray(action_stats[a1], dtype=np.float32)
        self.state_low = np.asarray(state_stats[s0], dtype=np.float32)
        self.state_high = np.asarray(state_stats[s1], dtype=np.float32)
        if self.action_low.shape[0] < ACTION_DIM or self.state_low.shape[0] < ACTION_DIM:
            raise ValueError(f"Stats dim must be >= {ACTION_DIM}")
        logging.info("Loaded stats key `%s` from %s", key, stats_json)

    def preprocess_state(self, state_raw: np.ndarray) -> np.ndarray:
        state_raw = _to_1d_float32(state_raw)
        left_joint = state_raw[0:6]
        left_gripper = state_raw[6]
        right_joint = state_raw[7:13]
        right_gripper = state_raw[13]
        state14 = np.zeros((ACTION_DIM,), dtype=np.float32)
        state14[0:6] = _normalize_values(left_joint, self.state_low[0:6], self.state_high[0:6], STATE_NORM_TYPE)
        state14[6] = _normalize_values(
            np.array([left_gripper], dtype=np.float32),
            np.array([self.state_low[12]], dtype=np.float32),
            np.array([self.state_high[12]], dtype=np.float32),
            STATE_NORM_TYPE,
        )[0]
        state14[7:13] = _normalize_values(right_joint, self.state_low[6:12], self.state_high[6:12], STATE_NORM_TYPE)
        state14[13] = _normalize_values(
            np.array([right_gripper], dtype=np.float32),
            np.array([self.state_low[13]], dtype=np.float32),
            np.array([self.state_high[13]], dtype=np.float32),
            STATE_NORM_TYPE,
        )[0]
        out = np.full((MODEL_STATE_DIM,), STATE_PAD_VALUE, dtype=np.float32)
        out[:ACTION_DIM] = state14
        return out

    def postprocess_actions(self, normalized_actions: np.ndarray, state_raw: np.ndarray) -> np.ndarray:
        actions = np.asarray(normalized_actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        state_raw = _to_1d_float32(state_raw)
        left_delta = _unnormalize_values(actions[:, 0:6], self.action_low[0:6], self.action_high[0:6], NORM_TYPE)
        left_gripper = _unnormalize_values(
            actions[:, 6:7],
            np.array([self.action_low[12]], dtype=np.float32),
            np.array([self.action_high[12]], dtype=np.float32),
            NORM_TYPE,
        )
        right_delta = _unnormalize_values(actions[:, 7:13], self.action_low[6:12], self.action_high[6:12], NORM_TYPE)
        right_gripper = _unnormalize_values(
            actions[:, 13:14],
            np.array([self.action_low[13]], dtype=np.float32),
            np.array([self.action_high[13]], dtype=np.float32),
            NORM_TYPE,
        )
        out = np.zeros_like(actions, dtype=np.float32)
        out[:, 0:6] = state_raw[0:6] + left_delta
        out[:, 6:7] = left_gripper
        out[:, 7:13] = state_raw[7:13] + right_delta
        out[:, 13:14] = right_gripper
        return out


class GripperPostprocessor:
    """Final gripper command shaping used before publishing robot commands."""

    MIN_TRAVEL = {6: 0.3, 13: 0.3}
    REOPEN_TRAVEL = 0.2

    def __init__(self, cfg: RuntimeConfig) -> None:
        self.boost = float(cfg.gripper_boost)
        self.boost_max = float(cfg.gripper_boost_max)
        self.close_threshold = float(cfg.gripper_close_threshold)
        self.reset()

    def reset(self) -> None:
        self._state = {
            6: {"base": None, "closed": False, "peak": None},
            13: {"base": None, "closed": False, "peak": None},
        }

    def apply(self, actions14: np.ndarray, chunk_index: int) -> np.ndarray:
        actions = np.asarray(actions14, dtype=np.float32).copy()
        if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
            return actions
        original = actions.copy()
        masks = {6: np.zeros(actions.shape[0], dtype=bool), 13: np.zeros(actions.shape[0], dtype=bool)}
        if self.boost > 0:
            for idx in (6, 13):
                masks[idx] = self._close_mask(idx, original[:, idx])
                actions[masks[idx], idx] = np.minimum(original[masks[idx], idx] + self.boost, self.boost_max)
        if chunk_index <= 3 or chunk_index % 10 == 0:
            logging.info(
                "gripper postprocess infer#%d boost=%.3f max=%.3f threshold=%.3f close(L,R)=%d/%d,%d/%d",
                chunk_index,
                self.boost,
                self.boost_max,
                self.close_threshold,
                int(np.sum(masks[6])),
                int(masks[6].size),
                int(np.sum(masks[13])),
                int(masks[13].size),
            )
        return actions

    def _close_mask(self, gripper_idx: int, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        mask = np.zeros(values.size, dtype=bool)
        state = self._state[gripper_idx]
        base = state["base"]
        peak = state["peak"]
        min_travel = self.MIN_TRAVEL[gripper_idx]
        for i, value in enumerate(values):
            if state["closed"]:
                peak = value if peak is None else max(float(peak), float(value))
                if peak is not None and value <= float(peak) - self.REOPEN_TRAVEL:
                    state["closed"] = False
                    base = value
                    peak = None
            if not state["closed"] and (base is None or value < base):
                base = value
            if base is not None and value - base >= min_travel:
                state["closed"] = True
                peak = value if peak is None else max(float(peak), float(value))
            mask[i] = bool(state["closed"]) or value > self.close_threshold
        state["base"] = base
        state["peak"] = peak
        return mask


class WebsocketPolicyClient:
    """Thin client for the deployed policy websocket server."""

    def __init__(self, host: str, port: int) -> None:
        self.uri = f"ws://{host}:{port}"
        self.ws = None
        self.connect()

    def connect(self) -> None:
        self.close()
        for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(name, None)
        logging.info("Connecting to %s", self.uri)
        self.ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None, open_timeout=30)
        metadata = unpackb(self.ws.recv())
        logging.info("Server metadata: %s", metadata)

    def _send(self, data: dict[str, Any]) -> dict[str, Any]:
        assert self.ws is not None
        self.ws.send(packb(data))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        decoded = unpackb(response)
        if not isinstance(decoded, dict):
            raise RuntimeError(f"Unexpected websocket response type: {type(decoded)}")
        if not decoded.get("ok", True):
            raise RuntimeError(str(decoded.get("error", decoded)))
        return decoded

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._send({"type": "infer", "payload": payload})
        except Exception:
            logging.warning("Websocket infer failed once, reconnecting...", exc_info=True)
            self.connect()
            return self._send({"type": "infer", "payload": payload})

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None


class ROS2Bridge:
    """ROS2 subscribers for 3 cameras and arm states, plus RobotCmd publishers."""

    def __init__(self, robot_cmd_type: type[Any], robot_status_type: type[Any]) -> None:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import Int32MultiArray

        self._rclpy = rclpy
        self._bridge = CvBridge()
        self._robot_cmd_type = robot_cmd_type
        self.head_image: np.ndarray | None = None
        self.left_image: np.ndarray | None = None
        self.right_image: np.ndarray | None = None
        self.left_controller_state: np.ndarray | None = None
        self.right_controller_state: np.ndarray | None = None
        self.left_feedback_state: np.ndarray | None = None
        self.right_feedback_state: np.ndarray | None = None
        self.last_joy = [0, 0, 0, 0]
        self.triggered_joys: dict[int, list[int]] = {}
        self.joy_lock = threading.Lock()

        self.node = Node("realworld_inference_ros2")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread: threading.Thread | None = None

        self.left_pub = self.node.create_publisher(robot_cmd_type, CMD_LEFT_TOPIC, 10)
        self.right_pub = self.node.create_publisher(robot_cmd_type, CMD_RIGHT_TOPIC, 10)
        self.node.create_subscription(CompressedImage, IMG_HEAD_TOPIC, self._head_cb, 2)
        self.node.create_subscription(CompressedImage, IMG_LEFT_TOPIC, self._left_cb, 2)
        self.node.create_subscription(CompressedImage, IMG_RIGHT_TOPIC, self._right_cb, 2)
        self.node.create_subscription(robot_status_type, MON_LEFT_TOPIC, self._controller_left_cb, 10)
        self.node.create_subscription(robot_status_type, MON_RIGHT_TOPIC, self._controller_right_cb, 10)
        self.node.create_subscription(robot_status_type, FB_LEFT_TOPIC, self._feedback_left_cb, 10)
        self.node.create_subscription(robot_status_type, FB_RIGHT_TOPIC, self._feedback_right_cb, 10)
        self.node.create_subscription(Int32MultiArray, JOY_TOPIC, self._joy_cb, 10)

    def start(self) -> None:
        self._spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin_thread.start()

    def close(self) -> None:
        try:
            self.executor.shutdown()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass

    def _decode_compressed(self, msg: Any) -> np.ndarray:
        arr = np.asarray(self._bridge.compressed_imgmsg_to_cv2(msg, "passthrough"))
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    def _head_cb(self, msg: Any) -> None:
        self.head_image = self._decode_compressed(msg)

    def _left_cb(self, msg: Any) -> None:
        self.left_image = self._decode_compressed(msg)

    def _right_cb(self, msg: Any) -> None:
        self.right_image = self._decode_compressed(msg)

    def _controller_left_cb(self, msg: Any) -> None:
        self.left_controller_state = np.asarray(msg.joint_pos, dtype=np.float32).copy()

    def _controller_right_cb(self, msg: Any) -> None:
        self.right_controller_state = np.asarray(msg.joint_pos, dtype=np.float32).copy()

    def _feedback_left_cb(self, msg: Any) -> None:
        self.left_feedback_state = np.asarray(msg.joint_pos, dtype=np.float32).copy()

    def _feedback_right_cb(self, msg: Any) -> None:
        self.right_feedback_state = np.asarray(msg.joint_pos, dtype=np.float32).copy()

    def _joy_cb(self, msg: Any) -> None:
        joy = list(msg.data)
        with self.joy_lock:
            for i in range(min(4, len(joy))):
                if self.last_joy[i] == 0 and joy[i] == 1:
                    self.triggered_joys[i] = joy.copy()
            self.last_joy = joy

    def pop_triggered_joys(self) -> dict[int, list[int]]:
        with self.joy_lock:
            triggered = dict(self.triggered_joys)
            self.triggered_joys.clear()
        return triggered

    def get_qpos(self) -> np.ndarray | None:
        left = self.left_controller_state if self.left_controller_state is not None else self.left_feedback_state
        right = self.right_controller_state if self.right_controller_state is not None else self.right_feedback_state
        if left is None or right is None:
            return None
        return np.concatenate([np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)], axis=0)

    def get_feedback_qpos(self) -> np.ndarray | None:
        if self.left_feedback_state is None or self.right_feedback_state is None:
            return None
        return np.concatenate([np.asarray(self.left_feedback_state, dtype=np.float32), np.asarray(self.right_feedback_state, dtype=np.float32)], axis=0)

    def read(self, timeout_sec: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        deadline = time.time() + timeout_sec
        while self._rclpy.ok() and time.time() < deadline:
            qpos = self.get_qpos()
            if self.head_image is not None and self.left_image is not None and self.right_image is not None and qpos is not None:
                return self.head_image.copy(), self.left_image.copy(), self.right_image.copy(), qpos
            time.sleep(0.01)
        return None

    def publish(self, action14: np.ndarray, mode: int = 5) -> None:
        action14 = _to_1d_float32(action14)
        left_msg = self._robot_cmd_type()
        left_msg.header.stamp = self.node.get_clock().now().to_msg()
        left_msg.joint_pos = np.asarray(action14[:6], dtype=np.float64)
        left_msg.gripper = float(action14[6])
        left_msg.mode = int(mode)
        self.left_pub.publish(left_msg)

        right_msg = self._robot_cmd_type()
        right_msg.header.stamp = self.node.get_clock().now().to_msg()
        right_msg.joint_pos = np.asarray(action14[7:13], dtype=np.float64)
        right_msg.gripper = float(action14[13])
        right_msg.mode = int(mode)
        self.right_pub.publish(right_msg)


def _wait_for_qpos(ros_bridge: ROS2Bridge, rclpy: Any, timeout_sec: float) -> np.ndarray | None:
    deadline = time.time() + timeout_sec
    while rclpy.ok() and time.time() < deadline:
        qpos = ros_bridge.get_feedback_qpos()
        if qpos is None:
            qpos = ros_bridge.get_qpos()
        if qpos is not None:
            return qpos
        time.sleep(0.01)
    return None


def _publish_reset(ros_bridge: ROS2Bridge, rclpy: Any) -> None:
    qpos = _wait_for_qpos(ros_bridge, rclpy, RESET_WAIT_STATE_TIMEOUT_SEC)
    if qpos is not None:
        max_diff = max(float(np.max(np.abs(RESET_ACTION[:6] - qpos[:6]))), float(np.max(np.abs(RESET_ACTION[7:13] - qpos[7:13]))))
        steps = max(1, int(np.ceil(max_diff / RESET_INTERP_MAX_STEP_RAD)))
        logging.info("reset interpolation max_diff=%.4f steps=%d", max_diff, steps)
        for step in range(1, steps + 1):
            alpha = step / float(steps)
            action = RESET_ACTION.copy()
            action[:6] = qpos[:6] + (RESET_ACTION[:6] - qpos[:6]) * alpha
            action[7:13] = qpos[7:13] + (RESET_ACTION[7:13] - qpos[7:13]) * alpha
            ros_bridge.publish(action)
            time.sleep(1.0 / RESET_INTERP_HZ)
    deadline = time.time() + RESET_HOLD_SEC
    while rclpy.ok() and time.time() < deadline:
        ros_bridge.publish(RESET_ACTION)
        time.sleep(1.0 / CONTROL_HZ)


def _restore_gravity(ros_bridge: ROS2Bridge) -> None:
    qpos = ros_bridge.get_qpos()
    if qpos is None:
        logging.warning("cannot switch to gravity compensation: no arm state")
        return
    ros_bridge.publish(qpos, mode=3)


def _build_policy_payload(
    *,
    temporal_views: list[np.ndarray],
    temporal_state: np.ndarray,
    instruction: str,
    unnorm_key: str,
    raw_examples: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    payload = {
        "batch_images": [[
            [image for image in temporal_views[0]],
            [image for image in temporal_views[1]],
            [image for image in temporal_views[2]],
        ]],
        "state": [temporal_state],
        "instructions": [instruction],
        "unnorm_key": unnorm_key,
    }
    if raw_examples:
        payload["raw_examples"] = raw_examples
    return payload


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Core realworld Pi05MEM keyframe inference runner.")
    parser.add_argument("--ws-host", default=os.environ.get("WS_HOST", "10.140.60.127"))
    parser.add_argument("--ws-port", type=int, default=int(os.environ.get("WS_PORT", "2333")))
    parser.add_argument("--stats-json", default=os.environ.get("STATS_JSON", "checkpoints/find_hidden_block/dataset_statistics.json"))
    parser.add_argument("--unnorm-key", default=os.environ.get("UNNORM_KEY", "a1_lift2"))
    parser.add_argument("--instruction", action="append", default=None, help="May be passed multiple times.")
    parser.add_argument("--temporal-frames", type=int, default=4)
    parser.add_argument("--temporal-stride", type=int, default=20)
    parser.add_argument("--no-temporal-anchor-first", action="store_true")
    parser.add_argument("--keyframes", type=int, default=int(os.environ.get("KEYFRAME_MEMORY_MAX_KEYFRAMES", "3")))
    parser.add_argument("--keyframe-threshold", type=float, default=float(os.environ.get("KEYFRAME_EVENT_COMMIT_THRESHOLD", "0.55")))
    parser.add_argument("--keyframe-min-gap", type=int, default=50)
    parser.add_argument("--keyframe-cooldown", type=int, default=50)
    parser.add_argument("--keyframe-merge-window", type=int, default=20)
    parser.add_argument("--gripper-boost", type=float, default=float(os.environ.get("GRIPPER_CLOSE_BOOST_OFFSET", "0.2")))
    parser.add_argument("--gripper-boost-max", type=float, default=float(os.environ.get("GRIPPER_CLOSE_BOOST_MAX", "0.0")))
    parser.add_argument(
        "--gripper-close-threshold",
        dest="gripper_close_threshold",
        type=float,
        default=float(os.environ.get("GRIPPER_CLOSE_THRESHOLD", "-2.6")),
        help="Gripper values above this threshold are treated as close commands.",
    )
    parser.add_argument("--debug-interval", type=int, default=int(os.environ.get("DEBUG_INTERVAL", "5")))
    parser.add_argument("--reset-on-start", action="store_true")
    args = parser.parse_args()
    instructions = args.instruction or [os.environ.get("INSTRUCTION", DEFAULT_INSTRUCTION)]
    instructions = [item.strip() for item in instructions if item and item.strip()]
    if not instructions:
        raise ValueError("at least one instruction is required")
    return RuntimeConfig(
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        stats_json=args.stats_json,
        unnorm_key=args.unnorm_key,
        instructions=instructions,
        temporal_frames=max(1, args.temporal_frames),
        temporal_stride=max(1, args.temporal_stride),
        temporal_anchor_first=not args.no_temporal_anchor_first,
        keyframes=max(0, args.keyframes),
        keyframe_threshold=args.keyframe_threshold,
        keyframe_min_gap=args.keyframe_min_gap,
        keyframe_cooldown=args.keyframe_cooldown,
        keyframe_merge_window=args.keyframe_merge_window,
        gripper_boost=args.gripper_boost,
        gripper_boost_max=args.gripper_boost_max,
        gripper_close_threshold=args.gripper_close_threshold,
        debug_interval=args.debug_interval,
        reset_on_start=args.reset_on_start,
    )


def main() -> None:
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    stats_path = Path(cfg.stats_json)
    if not stats_path.exists():
        raise FileNotFoundError(f"stats_json not found: {stats_path}")

    from arx5_arm_msg.msg import RobotCmd, RobotStatus
    import rclpy

    rclpy.init()
    stop_requested = threading.Event()

    def _handle_stop_signal(signum: int, _frame: Any) -> None:
        logging.info("received signal %s; stopping", signum)
        stop_requested.set()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        raise KeyboardInterrupt

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in old_handlers:
        signal.signal(sig, _handle_stop_signal)

    ros_bridge: ROS2Bridge | None = None
    ws_client: WebsocketPolicyClient | None = None
    try:
        ros_bridge = ROS2Bridge(robot_cmd_type=RobotCmd, robot_status_type=RobotStatus)
        ros_bridge.start()
        adapter = StatsAdapter(stats_json=str(stats_path), unnorm_key=cfg.unnorm_key)
        gripper = GripperPostprocessor(cfg)
        temporal = TemporalObservationBuffer(
            num_frames=cfg.temporal_frames,
            stride=cfg.temporal_stride,
            anchor_first=cfg.temporal_anchor_first,
        )
        memory = PredictedKeyframeMemory(cfg)
        ws_client = WebsocketPolicyClient(host=cfg.ws_host, port=cfg.ws_port)
        instruction_idx = 0
        action_queue: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=ACTION_CHUNK_SIZE)
        running = False
        infer_count = 0
        step_count = 0
        last_timeout_log = 0.0

        logging.info(
            "core deploy config: ws=%s:%d stats=%s unnorm_key=%s temporal=%d stride=%d keyframes=%d threshold=%.3f gripper_boost=%.3f",
            cfg.ws_host,
            cfg.ws_port,
            cfg.stats_json,
            cfg.unnorm_key,
            cfg.temporal_frames,
            cfg.temporal_stride,
            cfg.keyframes,
            cfg.keyframe_threshold,
            cfg.gripper_boost,
        )
        logging.info("instructions=%d current=%s", len(cfg.instructions), cfg.instructions[instruction_idx])
        logging.info("/arx_joy: button1=start, button2=pause, button3=next instruction, button4=reset")
        if cfg.reset_on_start:
            _publish_reset(ros_bridge, rclpy)

        while rclpy.ok() and not stop_requested.is_set():
            t0 = time.perf_counter()
            for button_idx in ros_bridge.pop_triggered_joys():
                if button_idx == JOY_START_BUTTON:
                    running = True
                    temporal.reset()
                    memory.reset()
                    gripper.reset()
                    action_queue.clear()
                    logging.info("start control: instruction=%s", cfg.instructions[instruction_idx])
                elif button_idx == JOY_PAUSE_BUTTON:
                    running = False
                    action_queue.clear()
                    _restore_gravity(ros_bridge)
                    logging.info("pause control; switched to gravity mode")
                elif button_idx == JOY_NEXT_INSTRUCTION_BUTTON:
                    instruction_idx = (instruction_idx + 1) % len(cfg.instructions)
                    logging.info("next instruction[%d/%d]: %s", instruction_idx + 1, len(cfg.instructions), cfg.instructions[instruction_idx])
                elif button_idx == JOY_RESET_BUTTON:
                    running = False
                    temporal.reset()
                    memory.reset()
                    gripper.reset()
                    action_queue.clear()
                    _publish_reset(ros_bridge, rclpy)
                    logging.info("reset buffers and robot pose")

            obs = ros_bridge.read(timeout_sec=0.05)
            if obs is None:
                now = time.time()
                if now - last_timeout_log > 2.0:
                    logging.warning("waiting for images and arm state")
                    last_timeout_log = now
                continue

            head_raw, left_raw, right_raw, state_raw = obs
            if not running:
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))
                continue

            state_norm32 = adapter.preprocess_state(state_raw)
            temporal.add(head_img=head_raw, left_img=left_raw, right_img=right_raw, state_vec=state_norm32)
            committed = memory.commit_due(current_step=temporal.current_step(), temporal_buffer=temporal)
            if committed:
                logging.info("committed keyframes=%s memory=%s pending=%s", committed, memory.memory_steps(), memory.pending_steps())

            if not action_queue and temporal.ready():
                temporal_views, temporal_state, temporal_indices = temporal.build_payload_tensors()
                raw_examples = memory.build_raw_examples()
                payload = _build_policy_payload(
                    temporal_views=temporal_views,
                    temporal_state=temporal_state,
                    instruction=cfg.instructions[instruction_idx],
                    unnorm_key=cfg.unnorm_key,
                    raw_examples=raw_examples,
                )
                infer_started = time.time()
                response = ws_client.infer(payload)
                infer_data = response["data"]
                normalized = _coerce_normalized_actions(infer_data["normalized_actions"])
                infer_count += 1
                raw_actions = adapter.postprocess_actions(normalized[:ACTION_CHUNK_SIZE], state_raw)[:, :ACTION_DIM]
                command_actions = gripper.apply(raw_actions, infer_count)
                scheduled, commit_step, confidence, reason = memory.register_prediction(
                    infer_data=infer_data,
                    base_step=temporal.current_step(),
                    action_chunk_len=int(command_actions.shape[0]),
                )
                latency = time.time() - infer_started
                keyframe_debug = _summarize_keyframe_infer_data(infer_data)
                logging.info(
                    "infer#%d %.3fs chunk=%d temporal=%s memory=%s pending=%s scheduled=%s reason=%s commit=%d conf=%.3f keyframe=%s",
                    infer_count,
                    latency,
                    command_actions.shape[0],
                    temporal_indices,
                    memory.memory_steps(),
                    memory.pending_steps(),
                    scheduled,
                    reason,
                    commit_step,
                    confidence,
                    keyframe_debug,
                )
                for idx, (cmd, raw) in enumerate(zip(command_actions, raw_actions)):
                    action_queue.append((np.asarray(cmd, dtype=np.float32), np.asarray(raw, dtype=np.float32)))

            if action_queue:
                command_action, raw_action = action_queue.popleft()
                ros_bridge.publish(command_action)
                step_count += 1
                if cfg.debug_interval > 0 and (step_count <= 3 or step_count % cfg.debug_interval == 0):
                    logging.info(
                        "step#%d queue=%d cmd_gripper=(%.3f, %.3f) raw_gripper=(%.3f, %.3f)",
                        step_count,
                        len(action_queue),
                        float(command_action[6]),
                        float(command_action[13]),
                        float(raw_action[6]),
                        float(raw_action[13]),
                    )
            else:
                logging.warning("empty action queue after inference")

            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        logging.info("interrupted")
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        if ws_client is not None:
            ws_client.close()
        if ros_bridge is not None:
            ros_bridge.close()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
