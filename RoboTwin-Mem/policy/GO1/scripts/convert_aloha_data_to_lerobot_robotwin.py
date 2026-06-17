"""
Modified from https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py

Script to convert Aloha hdf5 data to the LeRobot dataset v2.1 format.
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch
import re


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def load_episode_metadata(ep_path: Path) -> dict:
    metadata_path = ep_path.parent / "episode_info.json"
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f_meta:
        return json.load(f_meta)


def collect_hdf5_files(raw_dir: Path) -> list[Path]:
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(Path(root) / filename)

    def extract_episode_index(path: Path) -> int:
        for candidate in (path.parent.name, path.stem):
            match = re.search(r"episode[_-]?(\d+)", candidate)
            if match is not None:
                return int(match.group(1))
        raise ValueError(f"Cannot infer episode index from path: {path}")

    return sorted(hdf5_files, key=extract_episode_index)


def write_episode_metadata(repo_id: str, episode_metadata_by_episode: list[dict]):
    episodes_path = HF_LEROBOT_HOME / repo_id / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"LeRobot metadata not found at {episodes_path}")

    updated_lines = []
    with open(episodes_path, "r", encoding="utf-8") as f_epi:
        for episode_list_index, line in enumerate(f_epi):
            line = line.strip()
            if not line:
                continue

            episode_record = json.loads(line)
            if episode_list_index < len(episode_metadata_by_episode):
                episode_metadata = dict(episode_metadata_by_episode[episode_list_index])
                episode_metadata.pop("episode_index", None)
                episode_record.update(episode_metadata)
            updated_lines.append(json.dumps(episode_record, ensure_ascii=False))

    with open(episodes_path, "w", encoding="utf-8") as f_epi:
        f_epi.write("\n".join(updated_lines))
        f_epi.write("\n")


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> tuple[LeRobotDataset, list[dict]]:
    if episodes is None:
        episodes = range(len(hdf5_files))

    episode_metadata_by_episode = []

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        episode_metadata = load_episode_metadata(ep_path)
        num_frames = state.shape[0]
        dir_path = os.path.dirname(ep_path)
        json_Path = f"{dir_path}/instructions.json"

        with open(json_Path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict["instructions"]
            instruction = np.random.choice(instructions)
        for i in range(num_frames):
            frame = {"observation.state": state[i], "action": action[i]}

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            dataset.add_frame(frame, task=instruction)
        dataset.save_episode()
        episode_metadata_by_episode.append(episode_metadata)

    return dataset, episode_metadata_by_episode


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
    hdf5_files = collect_hdf5_files(raw_dir)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset, episode_metadata_by_episode = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    write_episode_metadata(repo_id, episode_metadata_by_episode)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
