import os
import h5py
import numpy as np
import cv2
import argparse
import yaml
import json


KEYFRAME_METADATA_KEYS = ("keyframe_steps",)


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def load_scene_info(path):
    scene_info_path = os.path.join(path, "scene_info.json")
    if not os.path.isfile(scene_info_path):
        return {}

    with open(scene_info_path, "r", encoding="utf-8") as f_info:
        return json.load(f_info)


def _map_keyframe_steps(raw_steps, raw_episode_length, processed_episode_length, episode_idx, key_name):
    if raw_steps is None:
        return []
    if processed_episode_length <= 0:
        if len(raw_steps) == 0:
            return []
        raise ValueError(
            f"episode_{episode_idx} has {key_name}={raw_steps}, but processed episode length is 0"
        )

    mapped_steps = []
    seen_steps = set()
    for raw_step in raw_steps:
        step = int(raw_step)
        if step < 0 or step >= raw_episode_length:
            raise ValueError(
                f"episode_{episode_idx} has raw {key_name} step {step}, "
                f"but raw episode length is {raw_episode_length}"
            )
        mapped_step = min(step, processed_episode_length - 1)
        if mapped_step not in seen_steps:
            mapped_steps.append(mapped_step)
            seen_steps.add(mapped_step)
    return mapped_steps


def get_episode_metadata(scene_info_db, episode_idx, raw_episode_length, processed_episode_length):
    episode_record = scene_info_db.get(f"episode_{episode_idx}", {})
    info_section = episode_record.get("info", {}) if isinstance(episode_record, dict) else {}

    episode_metadata = {"episode_index": int(episode_idx)}
    for key_name in KEYFRAME_METADATA_KEYS:
        raw_steps = None
        if isinstance(info_section, dict) and key_name in info_section:
            raw_steps = info_section.get(key_name)
        elif isinstance(episode_record, dict) and key_name in episode_record:
            raw_steps = episode_record.get(key_name)

        if raw_steps is not None:
            episode_metadata[key_name] = _map_keyframe_steps(
                raw_steps,
                raw_episode_length=raw_episode_length,
                processed_episode_length=processed_episode_length,
                episode_idx=episode_idx,
                key_name=key_name,
            )

    return episode_metadata


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    scene_info_db = load_scene_info(path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):

        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
            os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
            "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(
            os.path.join(path, "data", f"episode{i}.hdf5")
        )
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])
            state = state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        episode_metadata = get_episode_metadata(
            scene_info_db,
            episode_idx=i,
            raw_episode_length=int(left_gripper_all.shape[0]),
            processed_episode_length=len(qpos),
        )
        with open(
            os.path.join(save_path, f"episode_{i}", "episode_info.json"),
            "w",
            encoding="utf-8",
        ) as f_meta:
            json.dump(episode_metadata, f_meta, ensure_ascii=False, indent=2)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    data_root = os.environ.get("ROBOTWIN_MEM_DATA_DIR")
    if data_root:
        load_dir = os.path.join(data_root, str(task_name), str(setting))
    else:
        load_dir = os.path.join("../../data", str(task_name), str(setting))

    begin = 0
    print(f"read data from path:{load_dir}")

    processed_root = os.environ.get("ROBOTWIN_MEM_GO1_PROCESSED_DIR", "processed_data")
    target_dir = os.path.join(processed_root, f"{task_name}-{setting}-{expert_data_num}")
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
    )
