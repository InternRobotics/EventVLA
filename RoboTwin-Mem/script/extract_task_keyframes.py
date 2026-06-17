#!/usr/bin/env python3

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract start/keyframe/end images for selected RoboTwin-Mem tasks."
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
        help="Directory to save extracted figures.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task names to process.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to extract for every task.",
    )
    parser.add_argument(
        "--camera",
        default="cam_high",
        help="Camera name inside observations/images.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Upscale factor applied before saving PNGs for easier viewing.",
    )
    return parser.parse_args()


def load_episode_info(episode_dir: Path) -> dict:
    info_path = episode_dir / "episode_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_image(encoded_bytes):
    image = Image.open(io.BytesIO(encoded_bytes)).convert("RGB")
    image_array = np.asarray(image)
    image_rgb = image_array[..., [2, 1, 0]]
    return Image.fromarray(image_rgb)


def upscale_image(image: Image.Image, scale: int) -> Image.Image:
    if scale <= 1:
        return image
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)


def build_contact_sheet(images_with_labels, scale: int) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 22 * max(scale, 1)
    padding = 12 * max(scale, 1)
    columns = min(3, len(images_with_labels))
    rows = (len(images_with_labels) + columns - 1) // columns

    tile_width = max(image.width for image, _ in images_with_labels)
    tile_height = max(image.height for image, _ in images_with_labels)

    canvas = Image.new(
        "RGB",
        (
            columns * tile_width + (columns + 1) * padding,
            rows * (tile_height + label_height) + (rows + 1) * padding,
        ),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)

    for idx, (image, label) in enumerate(images_with_labels):
        row = idx // columns
        col = idx % columns
        x = padding + col * tile_width + col * padding
        y = padding + row * (tile_height + label_height) + row * padding
        canvas.paste(image, (x, y + label_height))
        draw.text((x, y), label, fill=(25, 25, 25), font=font)

    return canvas


def extract_task_frames(task: str, episode_idx: int, camera: str, scale: int, processed_root: Path, output_root: Path):
    task_root = processed_root / f"{task}-demo_clean-50"
    episode_dir = task_root / f"episode_{episode_idx}"
    hdf5_path = episode_dir / f"episode_{episode_idx}.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing episode file: {hdf5_path}")

    info = load_episode_info(episode_dir)
    keyframe_steps = [int(step) for step in info.get("keyframe_steps", [])]

    output_dir = output_root / task
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        image_ds = f[f"observations/images/{camera}"]
        last_idx = len(image_ds) - 1
        frame_indices = [0] + keyframe_steps + [last_idx]

        unique_indices = []
        seen = set()
        for frame_idx in frame_indices:
            clamped_idx = max(0, min(int(frame_idx), last_idx))
            if clamped_idx not in seen:
                unique_indices.append(clamped_idx)
                seen.add(clamped_idx)

        manifest = {
            "task": task,
            "episode_index": episode_idx,
            "camera": camera,
            "source_hdf5": str(hdf5_path),
            "available_frame_count": len(image_ds),
            "source_resolution": None,
            "saved_frames": [],
        }

        images_with_labels = []
        for order, frame_idx in enumerate(unique_indices):
            role = "keyframe"
            if order == 0:
                role = "start"
            elif frame_idx == last_idx:
                role = "end"

            image = decode_image(image_ds[frame_idx])
            manifest["source_resolution"] = [image.width, image.height]
            image = upscale_image(image, scale)

            file_name = f"{order:02d}_{role}_frame_{frame_idx:04d}.png"
            save_path = output_dir / file_name
            image.save(save_path, format="PNG", compress_level=0)

            label = f"{role} | frame {frame_idx}"
            images_with_labels.append((image, label))
            manifest["saved_frames"].append(
                {
                    "role": role,
                    "frame_index": frame_idx,
                    "path": str(save_path),
                }
            )

    summary = build_contact_sheet(images_with_labels, scale=scale)
    summary_path = output_dir / "summary.png"
    summary.save(summary_path, format="PNG", compress_level=0)
    manifest["summary_path"] = str(summary_path)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main():
    args = parse_args()
    processed_root = args.processed_root or args.repo_root / "policy" / "GO1" / "processed_data"
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = []
    for task in args.tasks:
        manifest = extract_task_frames(
            task=task,
            episode_idx=args.episode,
            camera=args.camera,
            scale=args.scale,
            processed_root=processed_root,
            output_root=output_root,
        )
        manifests.append(manifest)
        print(
            f"[ok] {task}: saved {len(manifest['saved_frames'])} frames "
            f"to {output_root / task}"
        )

    with (output_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(manifests, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
