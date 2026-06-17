#!/usr/bin/env python3

import os
import json
import shutil
import sys
import time
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_runtime_args(task_config: str, cache_root: Path):
    config_path = REPO_ROOT / "task_config" / f"{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = "cover_blocks_hard"
    args["task_config"] = task_config
    args["save_path"] = str(cache_root.resolve())
    args["need_plan"] = True
    args["save_data"] = True
    args["collect_data"] = False

    embodiment_type = args.get("embodiment")
    embodiment_config_path = REPO_ROOT / "task_config" / "_embodiment_config.yml"
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        robot_path = Path(robot_file)
        if not robot_path.is_absolute():
            robot_path = REPO_ROOT / robot_path
        return str(robot_path.resolve())

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def save_rgb(save_path: Path, rgb):
    from PIL import Image

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(save_path)


def load_collected_rgb(env, frame_idx: int):
    if frame_idx in env.selected_rgb_frames:
        return env.selected_rgb_frames[frame_idx]
    if frame_idx in env.recent_rgb_frames:
        return env.recent_rgb_frames[frame_idx]
    raise FileNotFoundError(f"Frame {frame_idx} was not kept in the in-memory RGB cache.")


def build_frame_requests(keyframe_steps: list[int], keyframe_count: int):
    third_keyframe = keyframe_steps[2]
    k_frame = max(0, third_keyframe - 30)

    frame_requests = [
        ("k_minus_30", max(0, k_frame - 30)),
        ("k_minus_15", max(0, k_frame - 15)),
        ("k", k_frame),
        ("start", 0),
    ]
    for idx, frame_idx in enumerate(keyframe_steps[:keyframe_count], start=1):
        frame_requests.append((f"keyframe_{idx}", frame_idx))
    return frame_requests, k_frame


def export_selected_frames(
    env,
    output_dir: Path,
    camera_name: str,
    seed: int,
    keyframe_count: int,
):
    last_idx = env.FRAME_IDX - 1
    if last_idx < 0:
        raise RuntimeError("No frames were cached for this trajectory.")

    keyframe_steps = [int(step) for step in env.keyframe_steps]
    if len(keyframe_steps) < keyframe_count:
        raise RuntimeError(
            f"Expected at least {keyframe_count} keyframes, got {len(keyframe_steps)}: {keyframe_steps}"
        )
    if len(keyframe_steps) < 3:
        raise RuntimeError(f"Expected at least 3 keyframes to define k, got: {keyframe_steps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_requests, k_frame = build_frame_requests(keyframe_steps, keyframe_count)

    manifest = {
        "task": "cover_blocks_hard",
        "seed": seed,
        "camera": camera_name,
        "last_frame_index": last_idx,
        "reference_keyframe_index": 3,
        "reference_keyframe_frame": keyframe_steps[2],
        "k_frame": k_frame,
        "k_definition": "third keyframe minus 30 frames",
        "keyframe_steps": keyframe_steps[:keyframe_count],
        "saved_frames": [],
    }

    for order, (role, frame_idx) in enumerate(frame_requests):
        rgb = load_collected_rgb(env, frame_idx)
        save_path = output_dir / f"{order:02d}_{role}_frame_{frame_idx:04d}.png"
        save_rgb(save_path, rgb)
        manifest["saved_frames"].append(
            {
                "role": role,
                "frame_index": int(frame_idx),
                "path": str(save_path),
            }
        )
        print(f"Saved {role} frame {frame_idx} to: {save_path}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest to: {manifest_path}")
    return manifest


def cleanup_cache(cache_root: Path):
    if cache_root.exists():
        shutil.rmtree(cache_root)


def run_single_trajectory(
    args,
    output_dir: Path,
    camera_name: str,
    start_seed: int | None,
    max_tries: int,
    keyframe_count: int,
    keep_cache: bool,
):
    from envs.cover_blocks_hard import cover_blocks_hard  # noqa: E402
    from envs.utils.create_actor import UnStableError  # noqa: E402

    class CoverBlocksHardFrameCollector(cover_blocks_hard):
        def setup_demo(self, **kwargs):
            self.capture_camera_name = kwargs.pop("capture_camera_name", "head_camera")
            self.selected_rgb_frames = {}
            self.recent_rgb_frames = OrderedDict()
            super().setup_demo(**kwargs)

        def _remember_rgb_frame(self, frame_idx, rgb):
            rgb = rgb.copy()
            if frame_idx == 0:
                self.selected_rgb_frames[frame_idx] = rgb

            self.recent_rgb_frames[frame_idx] = rgb
            while len(self.recent_rgb_frames) > 61:
                self.recent_rgb_frames.popitem(last=False)

        def _remember_keyframe_context(self, keyframe_idx):
            for offset in (60, 45, 30, 0):
                frame_idx = max(0, keyframe_idx - offset)
                if frame_idx in self.recent_rgb_frames:
                    self.selected_rgb_frames[frame_idx] = self.recent_rgb_frames[frame_idx].copy()

        def _take_picture(self):
            if not self.save_data:
                return

            self.language_annotation_cache += 1
            frame_idx = self.FRAME_IDX
            self._update_render()
            self.cameras.update_picture()
            rgb_by_camera = self.cameras.get_rgb()
            if self.capture_camera_name not in rgb_by_camera:
                available = sorted(rgb_by_camera.keys())
                raise KeyError(
                    f"Camera '{self.capture_camera_name}' not found. Available cameras: {available}"
                )

            rgb = rgb_by_camera[self.capture_camera_name]["rgb"]
            self._remember_rgb_frame(frame_idx, rgb)
            super()._after_save_frame(frame_idx)
            if getattr(self, "_tracked_best_keyframe_step", None) == frame_idx:
                self._remember_keyframe_context(frame_idx)
            self.FRAME_IDX += 1

    attempt_seed = 0 if start_seed is None else start_seed
    tries = 0
    cache_root = Path(args["save_path"])

    while tries < max_tries:
        if not keep_cache:
            cleanup_cache(cache_root)

        env = CoverBlocksHardFrameCollector()
        success = False
        exported = False
        try:
            print(f"Trying seed {attempt_seed} ...")
            env.setup_demo(
                now_ep_num=0,
                seed=attempt_seed,
                capture_camera_name=camera_name,
                **args,
            )
            env.play_once()
            success = (
                env.plan_success
                and env.check_success()
                and env.FRAME_IDX > 0
                and len(env.keyframe_steps) >= keyframe_count
            )
        except UnStableError as exc:
            print(f"Unstable scene at seed {attempt_seed}: {exc}")
        except Exception as exc:
            print(f"Unexpected error at seed {attempt_seed}: {exc}")
        else:
            if success:
                manifest = export_selected_frames(
                    env=env,
                    output_dir=output_dir,
                    camera_name=camera_name,
                    seed=attempt_seed,
                    keyframe_count=keyframe_count,
                )
                exported = True
                print(f"Trajectory succeeded with seed {attempt_seed}.")
                return attempt_seed, manifest
            print(f"Trajectory failed with seed {attempt_seed}.")
        finally:
            try:
                env.close_env()
            except Exception:
                pass
            if args.get("render_freq"):
                try:
                    env.viewer.close()
                except Exception:
                    pass
            if not keep_cache and (exported or not success):
                cleanup_cache(cache_root)

        tries += 1
        attempt_seed += 1
        time.sleep(0.3)

    raise RuntimeError(f"Failed to collect a valid cover_blocks_hard trajectory after {max_tries} tries.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=REPO_ROOT / "figure" / "cover_blocks_hard_selected_frames",
    )
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tries", type=int, default=200)
    parser.add_argument("--keyframe-count", type=int, default=4)
    parser.add_argument("--keep-cache", action="store_true")
    return parser.parse_args()


def main():
    cli_args = parse_args()

    from test_render import Sapien_TEST  # noqa: E402
    import torch.multiprocessing as mp  # noqa: E402

    Sapien_TEST()
    mp.set_start_method("spawn", force=True)

    cli_args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = cli_args.output_dir / ".capture_cache"

    runtime_args = load_runtime_args(
        task_config=cli_args.task_config,
        cache_root=cache_root,
    )
    run_single_trajectory(
        args=runtime_args,
        output_dir=cli_args.output_dir,
        camera_name=cli_args.camera,
        start_seed=cli_args.seed,
        max_tries=cli_args.max_tries,
        keyframe_count=cli_args.keyframe_count,
        keep_cache=cli_args.keep_cache,
    )


if __name__ == "__main__":
    main()
