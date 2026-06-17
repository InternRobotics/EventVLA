import argparse
import json
import os
import random
import re
from typing import Any, Dict, List

import yaml

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def resolve_data_root(default_data_root: str = "data") -> str:
    data_root = os.environ.get("ROBOTWIN_MEM_DATA_DIR") or default_data_root
    if os.path.isabs(data_root):
        return data_root
    return os.path.abspath(os.path.join(parent_directory, "../..", data_root))


def extract_placeholders(instruction: str) -> List[str]:
    """Extract all placeholders of the form {X} from an instruction."""
    return re.findall(r"{([^}]+)}", instruction)


def normalize_episode_params(episode_params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize scene-info keys by stripping optional surrounding braces."""
    if not isinstance(episode_params, dict):
        return {}
    return {str(key).strip("{}"): value for key, value in episode_params.items()}


def filter_instructions(instructions: List[str], episode_params: Dict[str, Any]) -> List[str]:
    """
    Keep instructions whose placeholders are a subset of the available episode
    parameters. This also allows task-level fixed descriptions with no
    placeholders, which is required for hard tasks that only add metadata such
    as keyframe steps.
    """
    filtered_instructions = []
    shuffled_instructions = list(instructions)
    random.shuffle(shuffled_instructions)

    available_params = set(normalize_episode_params(episode_params).keys())

    for instruction in shuffled_instructions:
        placeholders = set(extract_placeholders(instruction))
        if not placeholders or placeholders.issubset(available_params):
            filtered_instructions.append(instruction)

    return filtered_instructions


def resolve_placeholder_value(key: str, value: Any, *, use_unseen: bool) -> str:
    """
    Resolve placeholder values.
    - Object-description references use seen/unseen object descriptions.
    - Single lowercase-letter placeholders are treated as arm placeholders.
    - Non-string values are converted to strings.
    """
    if isinstance(value, str):
        if "\\" in value or "/" in value:
            json_path = os.path.join(parent_directory, "../objects_description", value + ".json")
            if not os.path.exists(json_path):
                print(f"\033[1mERROR: '{json_path}' looks like a description file, but does not exist.\033[0m")
                exit()

        json_path = os.path.join(parent_directory, "../objects_description", value + ".json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            if use_unseen and json_data.get("unseen"):
                description = random.choice(json_data["unseen"])
            else:
                description = random.choice(json_data.get("seen", []))
            return f"the {description}"

    if len(key) == 1 and "a" <= key <= "z":
        return f"the {value} arm"

    return str(value)


def replace_placeholders(instruction: str, episode_params: Dict[str, Any]) -> str:
    """Replace placeholders using seen object descriptions."""
    stripped_episode_params = normalize_episode_params(episode_params)

    for key in extract_placeholders(instruction):
        if key not in stripped_episode_params:
            continue
        placeholder = "{" + key + "}"
        value = resolve_placeholder_value(key, stripped_episode_params[key], use_unseen=False)
        instruction = instruction.replace(placeholder, value)

    return instruction


def replace_placeholders_unseen(instruction: str, episode_params: Dict[str, Any]) -> str:
    """Replace placeholders using unseen object descriptions when available."""
    stripped_episode_params = normalize_episode_params(episode_params)

    for key in extract_placeholders(instruction):
        if key not in stripped_episode_params:
            continue
        placeholder = "{" + key + "}"
        value = resolve_placeholder_value(key, stripped_episode_params[key], use_unseen=True)
        instruction = instruction.replace(placeholder, value)

    return instruction


def load_task_instructions(task_name: str) -> Dict[str, Any]:
    """Load the task instructions from the JSON file."""
    file_path = os.path.join(parent_directory, f"../task_instruction/{task_name}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    return task_data


def load_scene_info(task_name: str, setting: str, scene_info_path: str) -> Dict[str, Dict]:
    """Load the scene info from the JSON file in the data directory."""
    file_path = os.path.join(resolve_data_root(scene_info_path), task_name, setting, "scene_info.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
        return scene_data
    except FileNotFoundError:
        print(f"\033[1mERROR: Scene info file '{file_path}' not found.\033[0m")
        exit(1)
    except json.JSONDecodeError:
        print(f"\033[1mERROR: Scene info file '{file_path}' contains invalid JSON.\033[0m")
        exit(1)


def extract_episodes_from_scene_info(scene_info: Dict) -> List[Dict[str, Any]]:
    """Extract episode parameters from scene_info."""
    episodes = []
    for _, episode_data in scene_info.items():
        if "info" in episode_data:
            episodes.append(episode_data["info"])
        else:
            episodes.append({})
    return episodes


def save_episode_descriptions(task_name: str, setting: str, generated_descriptions: List[Dict[str, Any]]):
    """Save generated descriptions to output files."""
    output_dir = os.path.join(resolve_data_root(), task_name, setting, "instructions")
    os.makedirs(output_dir, exist_ok=True)

    for episode_desc in generated_descriptions:
        episode_index = episode_desc["episode_index"]
        output_file = os.path.join(output_dir, f"episode{episode_index}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seen": episode_desc.get("seen", []),
                    "unseen": episode_desc.get("unseen", []),
                },
                f,
                indent=2,
            )


def build_default_descriptions(task_data: Dict[str, Any]) -> List[str]:
    """Fallback descriptions used when placeholder filtering yields nothing."""
    full_description = str(task_data.get("full_description", "") or "").strip()
    return [full_description] if full_description else []


def generate_episode_descriptions(task_name: str, episodes: List[Dict[str, Any]], max_descriptions: int = 1000000):
    """
    Generate descriptions for episodes by replacing placeholders in instructions
    with parameter values. If an episode only carries auxiliary metadata such as
    keyframe annotations, static full-task descriptions are still emitted.
    """
    task_data = load_task_instructions(task_name)
    seen_instructions = task_data.get("seen", [])
    unseen_instructions = task_data.get("unseen", [])
    default_descriptions = build_default_descriptions(task_data)

    all_generated_descriptions = []

    for i, episode in enumerate(episodes):
        filtered_seen_instructions = filter_instructions(seen_instructions, episode)
        filtered_unseen_instructions = filter_instructions(unseen_instructions, episode)

        if not filtered_seen_instructions and default_descriptions:
            filtered_seen_instructions = list(default_descriptions)
        if not filtered_unseen_instructions and default_descriptions:
            filtered_unseen_instructions = list(default_descriptions)

        if not filtered_seen_instructions and not filtered_unseen_instructions:
            print(f"Episode {i}: No valid instructions found")
            all_generated_descriptions.append({
                "episode_index": i,
                "seen": [],
                "unseen": [],
            })
            continue

        seen_episode_descriptions = []
        while len(seen_episode_descriptions) < max_descriptions and filtered_seen_instructions:
            for instruction in filtered_seen_instructions:
                if len(seen_episode_descriptions) >= max_descriptions:
                    break
                seen_episode_descriptions.append(replace_placeholders(instruction, episode))
            break

        unseen_episode_descriptions = []
        while len(unseen_episode_descriptions) < max_descriptions and filtered_unseen_instructions:
            for instruction in filtered_unseen_instructions:
                if len(unseen_episode_descriptions) >= max_descriptions:
                    break
                unseen_episode_descriptions.append(replace_placeholders_unseen(instruction, episode))
            break

        all_generated_descriptions.append(
            {
                "episode_index": i,
                "seen": seen_episode_descriptions,
                "unseen": unseen_episode_descriptions,
            }
        )

    return all_generated_descriptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate episode descriptions by replacing placeholders")
    parser.add_argument(
        "task_name",
        type=str,
        help="Name of the task (JSON file name without extension)",
    )
    parser.add_argument(
        "setting",
        type=str,
        help="Setting name used to construct the data directory path",
    )
    parser.add_argument(
        "max_num",
        type=int,
        default=100,
        help="Maximum number of descriptions per episode",
    )

    args = parser.parse_args()
    setting_file = os.path.join(parent_directory, f"../../task_config/{args.setting}.yml")
    with open(setting_file, "r", encoding="utf-8") as f:
        args_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

    scene_info = load_scene_info(args.task_name, args.setting, args_dict["save_path"])
    episodes = extract_episodes_from_scene_info(scene_info)

    results = generate_episode_descriptions(args.task_name, episodes, args.max_num)
    save_episode_descriptions(args.task_name, args.setting, results)
    print("Successfully Saved Instructions")
