#!/usr/bin/env python3

import argparse
import io
import json
from pathlib import Path


DEFAULT_TASKS = [
    "cover_blocks_hard",
    "pick_the_unhidden_block",
    "find_seal_and_seal_stamp",
    "pick_objects_in_order",
    "put_back_block_hard",
    "press_button_keyframe",
    "rearrange_blocks_hard",
    "reproduce_route",
]

DEFAULT_TASK_EPISODE_OVERRIDES = {
    "find_seal_and_seal_stamp": 25,
    "press_button_keyframe": 15,
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract one trajectory per task and save every Nth frame."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="RoboTwin-Mem repo root.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Root directory containing GO1 processed_data task folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "figure",
        help="Root directory to save extracted figures.",
    )
    parser.add_argument(
        "--output-name",
        default="eight_tasks_stride5",
        help="Subdirectory name under output-root.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task names to process.",
    )
    parser.add_argument(
        "--task-config",
        default="demo_clean",
        help="Task config name used in processed_data folder names.",
    )
    parser.add_argument(
        "--expert-data-num",
        type=int,
        default=50,
        help="Expert data number used in processed_data folder names.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Default episode index to extract.",
    )
    parser.add_argument(
        "--task-episode",
        action="append",
        default=[],
        metavar="TASK=EPISODE",
        help=(
            "Override episode index for one task. "
            "Can be passed multiple times. Default includes find_seal_and_seal_stamp=25."
        ),
    )
    parser.add_argument(
        "--camera",
        default="cam_high",
        help="Camera name inside observations/images.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Save one frame every this many steps.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Upscale factor applied before saving PNGs.",
    )
    parser.add_argument(
        "--include-last",
        action="store_true",
        help="Also save the last frame if it is not aligned to stride.",
    )
    return parser.parse_args()


def parse_task_episode_overrides(raw_overrides):
    overrides = dict(DEFAULT_TASK_EPISODE_OVERRIDES)
    for raw_override in raw_overrides:
        if "=" not in raw_override:
            raise ValueError(f"Invalid --task-episode value '{raw_override}', expected TASK=EPISODE")
        task, episode = raw_override.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Invalid --task-episode value '{raw_override}', empty task name")
        overrides[task] = int(episode)
    return overrides


def decode_image(encoded_or_array):
    import numpy as np
    from PIL import Image

    if isinstance(encoded_or_array, np.ndarray) and encoded_or_array.ndim == 3:
        return Image.fromarray(encoded_or_array).convert("RGB")

    if hasattr(encoded_or_array, "tobytes"):
        encoded_or_array = encoded_or_array.tobytes()
    encoded_bytes = bytes(encoded_or_array).rstrip(b"\0")

    image = Image.open(io.BytesIO(encoded_bytes)).convert("RGB")
    image_array = np.asarray(image)
    image_rgb = image_array[..., [2, 1, 0]]
    return Image.fromarray(image_rgb)


def upscale_image(image, scale: int):
    from PIL import Image

    if scale <= 1:
        return image
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)


def build_frame_indices(frame_count: int, stride: int, include_last: bool):
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if frame_count <= 0:
        return []

    frame_indices = list(range(0, frame_count, stride))
    last_idx = frame_count - 1
    if include_last and frame_indices[-1] != last_idx:
        frame_indices.append(last_idx)
    return frame_indices


def task_hdf5_path(processed_root: Path, task: str, task_config: str, expert_data_num: int, episode_idx: int):
    task_dir = processed_root / f"{task}-{task_config}-{expert_data_num}"
    episode_dir = task_dir / f"episode_{episode_idx}"
    return episode_dir / f"episode_{episode_idx}.hdf5"


def extract_task_stride_frames(
    task: str,
    episode_idx: int,
    camera: str,
    stride: int,
    scale: int,
    include_last: bool,
    task_config: str,
    expert_data_num: int,
    processed_root: Path,
    output_root: Path,
):
    import h5py

    hdf5_path = task_hdf5_path(
        processed_root=processed_root,
        task=task,
        task_config=task_config,
        expert_data_num=expert_data_num,
        episode_idx=episode_idx,
    )
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing episode file: {hdf5_path}")

    output_dir = output_root / task
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*_frame_*.png"):
        old_png.unlink()

    with h5py.File(hdf5_path, "r") as f:
        dataset_path = f"observations/images/{camera}"
        if dataset_path not in f:
            available = sorted(f["observations/images"].keys())
            raise KeyError(
                f"Camera '{camera}' not found in {hdf5_path}. Available cameras: {available}"
            )

        image_ds = f[dataset_path]
        frame_indices = build_frame_indices(len(image_ds), stride=stride, include_last=include_last)

        manifest = {
            "task": task,
            "episode_index": episode_idx,
            "camera": camera,
            "stride": stride,
            "include_last": include_last,
            "source_hdf5": str(hdf5_path),
            "available_frame_count": len(image_ds),
            "saved_frame_count": 0,
            "source_resolution": None,
            "saved_frames": [],
        }

        for order, frame_idx in enumerate(frame_indices):
            image = decode_image(image_ds[frame_idx])
            manifest["source_resolution"] = [image.width, image.height]
            image = upscale_image(image, scale)

            save_path = output_dir / f"{order:04d}_frame_{frame_idx:06d}.png"
            image.save(save_path, format="PNG", compress_level=0)
            manifest["saved_frames"].append(
                {
                    "order": order,
                    "frame_index": int(frame_idx),
                    "path": str(save_path),
                }
            )

    manifest["saved_frame_count"] = len(manifest["saved_frames"])
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main():
    args = parse_args()
    processed_root = args.processed_root or args.repo_root / "policy" / "GO1" / "processed_data"
    output_root = args.output_root / args.output_name
    output_root.mkdir(parents=True, exist_ok=True)
    task_episode_overrides = parse_task_episode_overrides(args.task_episode)

    manifests = []
    for task in args.tasks:
        episode_idx = task_episode_overrides.get(task, args.episode)
        manifest = extract_task_stride_frames(
            task=task,
            episode_idx=episode_idx,
            camera=args.camera,
            stride=args.stride,
            scale=args.scale,
            include_last=args.include_last,
            task_config=args.task_config,
            expert_data_num=args.expert_data_num,
            processed_root=processed_root,
            output_root=output_root,
        )
        manifests.append(manifest)
        print(
            f"[ok] {task} episode {episode_idx}: saved {manifest['saved_frame_count']} frames "
            f"to {output_root / task}"
        )

    index = {
        "tasks": args.tasks,
        "default_episode_index": args.episode,
        "task_episode_overrides": task_episode_overrides,
        "camera": args.camera,
        "stride": args.stride,
        "output_root": str(output_root),
        "manifests": manifests,
    }
    with (output_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote index to {output_root / 'index.json'}")


if __name__ == "__main__":
    main()
