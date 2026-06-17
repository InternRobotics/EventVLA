from copy import deepcopy

from ._base_task import Base_Task
from .utils import *


class cover_blocks_hard(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.num_slots = 4
        self.block_half_size = 0.02
        self.cover_x = [-0.33, -0.11, 0.11, 0.33]
        self.block_y = -0.14
        self.reveal_y_offset = 0.11
        self.cover_name = ["leftmost", "left-middle", "right-middle", "rightmost"]
        self.color_names = ["red", "green", "blue", "yellow"]
        self.color_tuple = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
        self.cover_init_quat = [0.5, 0.5, 0.5, 0.5]
        self.quat_of_target_pose = [0.0, 1.0, 0.0, 0.0]
        self.cover_pose_lst = []
        block_pose_lst = []

        for x_pos in self.cover_x:
            block_pose = rand_pose(
                xlim=[x_pos, x_pos],
                ylim=[self.block_y, self.block_y],
                zlim=[0.741 + self.block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            block_pose_lst.append(deepcopy(block_pose))

            cover_pose = rand_pose(
                xlim=[x_pos, x_pos],
                ylim=[self.block_y, self.block_y],
                zlim=[0.741, 0.741],
                qpos=self.cover_init_quat,
                rotate_rand=False,
            )
            self.cover_pose_lst.append(deepcopy(cover_pose))

        def create_cover(cover_pose):
            return create_actor(
                self,
                pose=cover_pose,
                modelname="003_cover",
                model_id=0,
                convex=True,
            )

        self.covers = [create_cover(cover_pose) for cover_pose in self.cover_pose_lst]

        def create_block(block_pose, color, color_name):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=color,
                name=f"box_{color_name}",
            )

        block_id_list = np.random.permutation(list(range(self.num_slots))).tolist()
        self.block_color_names = []
        self.blocks = []
        for slot_id, block_pose in enumerate(block_pose_lst):
            color_id = block_id_list[slot_id]
            color_name = self.color_names[color_id]
            self.block_color_names.append(color_name)
            self.blocks.append(create_block(block_pose, self.color_tuple[color_id], color_name))

        self.color_order_slots = [block_id_list.index(color_id) for color_id in range(self.num_slots)]
        self.cover_target_pose_xy = [[block_pose.p[0], block_pose.p[1]] for block_pose in block_pose_lst]
        self.reveal_pose_xy = [
            [block_pose.p[0], block_pose.p[1] + self.reveal_y_offset] for block_pose in block_pose_lst
        ]

        self.cover_dist_threshold = 0.03
        self.reveal_dist_threshold = 0.035
        self.cover_z_threshold = 0.742
        self.inspect_lift_height = 0.08
        self.inspect_delay = 2

        self.phase = "inspect"
        self.inspect_pointer = 0
        self.inspect_reveal_slot = None
        self.final_pointer = 0
        self.fail_flag = False
        self.reward_list = [0, 0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80, 1.0]
        self.keyframe_steps = []
        self._tracked_keyframe_slot = None
        self._tracked_best_keyframe_step = None
        self._tracked_best_keyframe_dist = float("-inf")

    def _progress_index(self):
        return self.inspect_pointer + self.final_pointer

    def _update_reward(self):
        self.max_reward = max(self.max_reward, self.reward_list[self._progress_index()])

    def _slot_arm(self, slot_id):
        return ArmTag("left" if self.cover_target_pose_xy[slot_id][0] < 0 else "right")

    def _target_pose(self, slot_id, reveal=False):
        target_xy = self.reveal_pose_xy[slot_id] if reveal else self.cover_target_pose_xy[slot_id]
        return target_xy + [0.741] + self.quat_of_target_pose

    def _start_keyframe_tracking(self, slot_id):
        if not self.save_data:
            return

        self._tracked_keyframe_slot = slot_id
        self._tracked_best_keyframe_step = None
        self._tracked_best_keyframe_dist = float("-inf")

    def _finish_keyframe_tracking(self, slot_id):
        if self._tracked_keyframe_slot != slot_id:
            return None

        keyframe_step = self._tracked_best_keyframe_step
        self._tracked_keyframe_slot = None
        self._tracked_best_keyframe_step = None
        self._tracked_best_keyframe_dist = float("-inf")
        return keyframe_step

    def _after_save_frame(self, frame_idx):
        if self._tracked_keyframe_slot is None:
            return

        slot_id = self._tracked_keyframe_slot
        cover_xy = np.asarray(self.covers[slot_id].get_pose().p[:2], dtype=np.float32)
        block_xy = np.asarray(self.blocks[slot_id].get_pose().p[:2], dtype=np.float32)
        dist_xy = float(np.linalg.norm(cover_xy - block_xy))
        if dist_xy > self._tracked_best_keyframe_dist + 1e-8:
            self._tracked_best_keyframe_dist = dist_xy
            self._tracked_best_keyframe_step = int(frame_idx)

    def get_keyframe_oracle_info(self):
        segment_active = False
        segment_slot = None
        dist_xy = None
        expected_slot = int(self.inspect_pointer) if 0 <= self.inspect_pointer < self.num_slots else None
        segment_candidates = []

        if self.phase == "inspect":
            slot_dist_records = []
            for slot_id in range(self.num_slots):
                slot_status = self._cover_status(slot_id)
                if slot_status == "covered":
                    continue

                cover_xy = np.asarray(self.covers[slot_id].get_pose().p[:2], dtype=np.float32)
                block_xy = np.asarray(self.blocks[slot_id].get_pose().p[:2], dtype=np.float32)
                slot_dist = float(np.linalg.norm(cover_xy - block_xy))
                slot_priority = 1 if (expected_slot is not None and slot_id == expected_slot) else 0
                slot_dist_records.append((slot_priority, slot_dist, slot_id))

            if len(slot_dist_records) > 0:
                slot_dist_records.sort(key=lambda item: (item[0], item[1]), reverse=True)
                _, dist_xy, segment_slot = slot_dist_records[0]
                segment_active = True
                segment_candidates = [
                    {
                        "slot": int(slot_id),
                        "d_xy": float(slot_dist),
                        "priority": int(slot_priority),
                    }
                    for slot_priority, slot_dist, slot_id in slot_dist_records
                ]

        return {
            "task_name": "cover_blocks_hard",
            "phase": str(self.phase),
            "inspect_pointer": int(self.inspect_pointer),
            "expected_slot": None if expected_slot is None else int(expected_slot),
            "fail_flag": bool(self.fail_flag),
            "segment_active": bool(segment_active),
            "segment_slot": None if segment_slot is None else int(segment_slot),
            "d_xy": None if dist_xy is None else float(dist_xy),
            "segment_candidates": segment_candidates,
            "env_step": int(getattr(self, "take_action_cnt", 0)),
        }

    def _cover_status(self, slot_id):
        cover_pose = self.covers[slot_id].get_pose().p
        dist_to_cover = np.linalg.norm(cover_pose[:2] - np.array(self.cover_target_pose_xy[slot_id]))
        dist_to_reveal = np.linalg.norm(cover_pose[:2] - np.array(self.reveal_pose_xy[slot_id]))

        if dist_to_cover < self.cover_dist_threshold and cover_pose[2] < self.cover_z_threshold:
            return "covered"
        if dist_to_reveal < self.reveal_dist_threshold:
            return "revealed"
        return "other"

    def _is_slot_open_for_progress(self, slot_id):
        status = self._cover_status(slot_id)
        if status == "revealed":
            return True

        cover_xy = np.asarray(self.covers[slot_id].get_pose().p[:2], dtype=np.float32)
        reveal_xy = np.asarray(self.reveal_pose_xy[slot_id], dtype=np.float32)
        dist_to_reveal = float(np.linalg.norm(cover_xy - reveal_xy))
        return dist_to_reveal < (self.reveal_dist_threshold * 1.25)

    def _mark_inspection_complete(self, slot_id):
        if self.fail_flag:
            return
        if self.phase != "inspect" or slot_id != self.inspect_pointer:
            self.fail_flag = True
            return

        self.inspect_pointer += 1
        self.inspect_reveal_slot = None
        if self.inspect_pointer == self.num_slots:
            self.phase = "final"
        self._update_reward()

    def _mark_final_open(self, slot_id):
        if self.fail_flag:
            return
        if self.phase != "final" or self.final_pointer >= self.num_slots:
            self.fail_flag = True
            return

        expected_slot = self.color_order_slots[self.final_pointer]
        if slot_id != expected_slot:
            self.fail_flag = True
            return

        self.final_pointer += 1
        self._update_reward()

    def update_state_transition(self):
        if self.fail_flag:
            return

        if self.phase == "inspect":
            if self.inspect_pointer >= self.num_slots:
                self.phase = "final"
            else:
                expected_slot = self.inspect_pointer
                is_expected_open = self._is_slot_open_for_progress(expected_slot)
                if self.inspect_reveal_slot is None:
                    if is_expected_open:
                        self.inspect_reveal_slot = expected_slot
                else:
                    if not is_expected_open:
                        self._mark_inspection_complete(expected_slot)

        if self.phase == "final" and self.final_pointer < self.num_slots:
            expected_final_slot = self.color_order_slots[self.final_pointer]
            opened_prefix_slots = set(self.color_order_slots[: self.final_pointer])

            for slot_id in range(self.num_slots):
                if slot_id == expected_final_slot or slot_id in opened_prefix_slots:
                    continue
                if self._is_slot_open_for_progress(slot_id):
                    self.fail_flag = True
                    return

            if self._is_slot_open_for_progress(expected_final_slot):
                self._mark_final_open(expected_final_slot)

    def play_once(self):
        self.keyframe_steps = []
        for slot_id in range(self.num_slots):
            self.inspect_cover(slot_id)
            if not self.plan_success or self.fail_flag:
                break

        if self.plan_success and not self.fail_flag:
            for slot_id in self.color_order_slots:
                self.final_open_cover(slot_id)
                if not self.plan_success or self.fail_flag:
                    break

        self.info["info"] = {
            "keyframe_steps": list(self.keyframe_steps),
        }
        return self.info

    def inspect_cover(self, slot_id):
        arm_tag = self._slot_arm(slot_id)
        instruction = (
            f"Open the {self.cover_name[slot_id]} cover to inspect the block color, then place the cover back."
        )

        self._start_keyframe_tracking(slot_id)
        keyframe_step = None
        try:
            self.move(
                self.grasp_actor(self.covers[slot_id], arm_tag=arm_tag, pre_grasp_dis=0.05),
                language_annotation=instruction,
            )
            self.move(
                self.move_by_displacement(arm_tag=arm_tag, z=self.inspect_lift_height),
                language_annotation=instruction,
            )
            self.move(
                self.move_by_displacement(arm_tag=arm_tag, y=self.reveal_y_offset),
                language_annotation=instruction,
            )
            self.delay(delay_time=self.inspect_delay, language_annotation=instruction)
            self.move(
                self.move_by_displacement(arm_tag=arm_tag, y=-self.reveal_y_offset),
                language_annotation=instruction,
            )
            self.move(
                self.place_actor(
                    self.covers[slot_id],
                    target_pose=self._target_pose(slot_id, reveal=False),
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.05,
                    dis=0.005,
                ),
                language_annotation=instruction,
            )
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05), language_annotation=instruction)
            self.move(self.back_to_origin(arm_tag=arm_tag), language_annotation=instruction)
        finally:
            keyframe_step = self._finish_keyframe_tracking(slot_id)

        if self.plan_success:
            if keyframe_step is not None:
                self.keyframe_steps.append(keyframe_step)
            self._mark_inspection_complete(slot_id)

    def final_open_cover(self, slot_id):
        arm_tag = self._slot_arm(slot_id)
        instruction = f"Finally open the cover hiding the {self.block_color_names[slot_id]} block."

        self.move(
            self.grasp_actor(self.covers[slot_id], arm_tag=arm_tag, pre_grasp_dis=0.05),
            language_annotation=instruction,
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05), language_annotation=instruction)
        self.move(
            self.place_actor(
                self.covers[slot_id],
                target_pose=self._target_pose(slot_id, reveal=True),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.005,
            ),
            language_annotation=instruction,
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.03), language_annotation=instruction)
        self.move(self.back_to_origin(arm_tag=arm_tag), language_annotation=instruction)

        if self.plan_success:
            self._mark_final_open(slot_id)

    def check_success(self):
        self.update_state_transition()

        if self.final_pointer == self.num_slots:
            self.max_reward = max(self.max_reward, 1.0)
            return True

        if self.fail_flag:
            return False
        return False
