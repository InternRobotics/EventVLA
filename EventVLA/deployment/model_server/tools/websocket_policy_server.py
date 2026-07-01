# Copyright 2025 eventvla community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import inspect
import logging
import traceback
import time

import numpy as np
import torch
from PIL import Image

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy
from . import image_tools


def _mask_value_to_bool(value) -> bool:
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


def _sample_images_mask(images_mask, sample_idx: int, view_count: int) -> list[bool]:
    if images_mask is None:
        return [True] * view_count
    try:
        sample_mask = images_mask[sample_idx]
    except Exception:
        return [True] * view_count

    if isinstance(sample_mask, np.ndarray):
        values = sample_mask.reshape(-1).tolist()
    elif isinstance(sample_mask, (list, tuple)):
        values = list(sample_mask)
    else:
        values = [sample_mask]

    return [
        _mask_value_to_bool(values[idx]) if idx < len(values) else True
        for idx in range(view_count)
    ]


def _blank_like_view(view):
    if isinstance(view, list):
        return [_blank_like_view(frame) for frame in view]
    if isinstance(view, tuple):
        return tuple(_blank_like_view(frame) for frame in view)
    if isinstance(view, Image.Image):
        return Image.new(view.mode, view.size, 0)
    return np.zeros_like(np.asarray(view))


def _to_pil_nested(images):
    if isinstance(images, np.ndarray) and images.ndim > 3:
        return [_to_pil_nested(item) for item in images]
    if isinstance(images, list):
        return [_to_pil_nested(item) for item in images]
    if isinstance(images, tuple):
        return tuple(_to_pil_nested(item) for item in images)
    return image_tools.to_pil_preserve(images)

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # 新增参数，单位秒，-1表示永不关闭
        metadata: dict | None = None,
        
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        """监控空闲时间，超时则关闭服务器"""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # 每次收到消息刷新活跃时间
                ret = self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer
        payload = msg.get("payload", msg)         # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # reset memory
        elif mtype in {"reset", "reset_memory"}:
            try:
                if hasattr(self._policy, "reset_memory"):
                    self._policy.reset_memory()
                elif hasattr(self._policy, "reset_memory_by_mask"):
                    # Default to single-sample reset.
                    self._policy.reset_memory_by_mask(reset_mask=torch.tensor([True], dtype=torch.bool))
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "reset_memory",
                    "request_id": req_id,
                }
            except Exception as e:
                logging.exception("Policy reset error (request_id=%s)", req_id)
                logging.exception(e)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset_memory",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))}
                }
            try:
                payload = self._prepare_predict_payload(payload)
                ouput_dict = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)
                
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                        # "traceback": traceback.format_exc(),
                    },
                }
            data = ouput_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }

    def _policy_uses_batch_images(self) -> bool:
        try:
            return "batch_images" in inspect.signature(self._policy.predict_action).parameters
        except (TypeError, ValueError):
            return False

    def _prepare_predict_payload(self, payload: dict) -> dict:
        payload = dict(payload)
        if self._policy_uses_batch_images() and "batch_images" not in payload and "examples" in payload:
            examples = list(payload["examples"])
            payload["batch_images"] = [example["image"] for example in examples]
            payload["instructions"] = [example["lang"] for example in examples]
            if "raw_examples" not in payload:
                payload["raw_examples"] = examples
            states = [example.get("state", None) for example in examples]
            if all(state is not None for state in states):
                payload["state"] = states

        if self._policy_uses_batch_images() and "batch_images" in payload:
            payload["batch_images"], payload["images_mask"] = self._prepare_batch_images_and_masks(
                payload["batch_images"],
                payload.get("images_mask"),
            )
            payload["run_eval"] = True
        return payload

    def _prepare_batch_images_and_masks(self, batch_images, images_mask):
        batch_images = _to_pil_nested(batch_images)
        expected_views = len(getattr(self._policy, "anchor_image_keys", ())) or 3

        padded_images = []
        padded_masks = []
        for sample_idx, sample in enumerate(batch_images):
            if isinstance(sample, (list, tuple)):
                sample = list(sample)
            else:
                sample = [sample]
            view_count = len(sample)
            if view_count > expected_views:
                raise ValueError(f"Expected at most {expected_views} views, got {view_count}")

            sample_mask = _sample_images_mask(images_mask, sample_idx, view_count)
            pad_count = expected_views - view_count
            if pad_count:
                if sample:
                    blank = _blank_like_view(sample[0])
                else:
                    blank = np.zeros((224, 224, 3), dtype=np.uint8)
                sample.extend([blank for _ in range(pad_count)])
                sample_mask.extend([False] * pad_count)

            padded_images.append(sample)
            padded_masks.append(sample_mask)
        return padded_images, padded_masks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
