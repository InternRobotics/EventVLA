"""Dataset mixture definitions for EventVLA."""

from typing import Dict, List, Tuple


DATASET_NAMED_MIXTURES: Dict[str, List[Tuple[str, float, str]]] = {
    "robotwin_mem": [
        ("cover_blocks_hard", 1.0, "robotwin_mem"),
        ("put_back_block_hard", 1.0, "robotwin_mem"),
        ("rearrange_blocks_hard", 1.0, "robotwin_mem"),
        ("observe_and_pickup_hard", 1.0, "robotwin_mem"),
        ("find_seal_and_seal_stamp", 1.0, "robotwin_mem"),
        ("observe_and_pickup_object", 1.0, "robotwin_mem"),
        ("reproduct_route", 1.0, "robotwin_mem"),
        ("press_button_keyframe", 1.0, "robotwin_mem"),
    ],
    "robotwin_mem8": [
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/cover_blocks_hard", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/find_seal_and_seal_stamp", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/pick_objects_in_order", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/pick_the_unhidden_block", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/press_button_keyframe", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/put_back_block_hard", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/rearrange_blocks_hard", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/reproduce_route", 1.0, "robotwin_mem"),
    ], 
    "robotwin_mem9": [
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/cover_blocks_hard", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/find_seal_and_seal_stamp", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/pick_objects_in_order", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/pick_the_unhidden_block", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/press_button_keyframe", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/put_back_block_hard", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/rearrange_blocks_hard", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata/reproduce_route", 1.0, "robotwin_mem"),
        ("/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/data/feiqi_data/swap_T_hard", 1.0, "robotwin_mem"),
    ], 
}
