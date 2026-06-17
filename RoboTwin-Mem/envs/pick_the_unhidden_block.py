from copy import deepcopy

from ._base_task import Base_Task
from .utils import *


class pick_the_unhidden_block(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.num_reference_slots = 3
        self.num_visible_blocks = 4

        self.block_half_size = 0.02

        self.cover_x = [-0.33, -0.11, 0.11]
        self.block_y = -0.12
        self.reveal_y_offset = 0.08

        self.visible_block_x = [-0.28, -0.1, 0.1, 0.28]
        self.visible_block_y = -0.24

        self.cover_name = ["left", "middle", "right"]
        self.color_names = ["red", "green", "blue", "yellow"]
        self.color_tuple = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

        self.cover_init_quat = [0.5, 0.5, 0.5, 0.5]
        self.quat_of_target_pose = [0.0, 1.0, 0.0, 0.0]

        self.cover_dist_threshold = 0.03
        self.reveal_dist_threshold = 0.035
        self.cover_z_threshold = 0.742

        self.inspect_lift_height = 0.08
        self.inspect_delay = 2

        # 成功判定：外部目标方块抬起约 1cm 即成功
        self.pickup_success_lift_height = 0.01
        self.pickup_success_tolerance = 0.002

        # demo 实际抬高要足够稳定，不能只抬 1cm
        self.pickup_demo_lift_height = 0.10

        self.cover_pose_lst = []
        covered_block_pose_lst = []
        visible_block_pose_lst = []

        for x_pos in self.cover_x:
            covered_block_pose = rand_pose(
                xlim=[x_pos, x_pos],
                ylim=[self.block_y, self.block_y],
                zlim=[0.741 + self.block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            covered_block_pose_lst.append(deepcopy(covered_block_pose))

            cover_pose = rand_pose(
                xlim=[x_pos, x_pos],
                ylim=[self.block_y, self.block_y],
                zlim=[0.741, 0.741],
                qpos=self.cover_init_quat,
                rotate_rand=False,
            )
            self.cover_pose_lst.append(deepcopy(cover_pose))

        for x_pos in self.visible_block_x:
            visible_block_pose = rand_pose(
                xlim=[x_pos, x_pos],
                ylim=[self.visible_block_y, self.visible_block_y],
                zlim=[0.741 + self.block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            visible_block_pose_lst.append(deepcopy(visible_block_pose))

        def create_cover(cover_pose):
            return create_actor(
                self,
                pose=cover_pose,
                modelname="003_cover",
                model_id=0,
                convex=True,
            )

        def create_block(block_pose, color, color_name, name_prefix):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=color,
                name=f"{name_prefix}_{color_name}",
            )

        self.covers = [create_cover(cover_pose) for cover_pose in self.cover_pose_lst]

        # 外部四个 visible blocks：四种颜色随机排列
        self.visible_block_color_ids = np.random.permutation(
            list(range(self.num_visible_blocks))
        ).tolist()

        self.visible_blocks = []
        self.visible_block_init_z = []

        for block_id, block_pose in enumerate(visible_block_pose_lst):
            color_id = self.visible_block_color_ids[block_id]
            color_name = self.color_names[color_id]

            visible_block = create_block(
                block_pose,
                self.color_tuple[color_id],
                color_name,
                "visible_block",
            )
            self.visible_blocks.append(visible_block)
            self.visible_block_init_z.append(float(block_pose.p[2]))

        # cover 内部只出现 3 种颜色
        # 外部 visible blocks 中没在 cover 内出现的那个颜色是目标
        self.target_color_id = int(np.random.randint(0, self.num_visible_blocks))

        covered_color_ids = [
            color_id
            for color_id in range(self.num_visible_blocks)
            if color_id != self.target_color_id
        ]
        np.random.shuffle(covered_color_ids)

        self.covered_block_color_ids = covered_color_ids
        self.covered_blocks = []

        for slot_id, block_pose in enumerate(covered_block_pose_lst):
            color_id = self.covered_block_color_ids[slot_id]
            color_name = self.color_names[color_id]

            covered_block = create_block(
                block_pose,
                self.color_tuple[color_id],
                color_name,
                "covered_block",
            )
            self.covered_blocks.append(covered_block)

        self.target_visible_block_id = self.visible_block_color_ids.index(
            self.target_color_id
        )

        self.cover_target_pose_xy = [
            [block_pose.p[0], block_pose.p[1]]
            for block_pose in covered_block_pose_lst
        ]

        self.reveal_pose_xy = [
            [block_pose.p[0], block_pose.p[1] + self.reveal_y_offset]
            for block_pose in covered_block_pose_lst
        ]

        self.phase = "inspect"
        self.inspect_pointer = 0
        self.inspect_reveal_slot = None
        self.pickup_done = False
        self.fail_flag = False

        self.reward_list = [0.0, 0.15, 0.30, 0.45, 1.0]
        self.keyframe_steps = []

        # 关键帧：cover 与内部方块 XY 距离最大的那一帧
        self._tracked_keyframe_slot = None
        self._tracked_best_keyframe_step = None
        self._tracked_best_keyframe_dist = float("-inf")

    def _progress_index(self):
        return self.inspect_pointer + int(self.pickup_done)

    def _update_reward(self):
        self.max_reward = max(
            self.max_reward,
            self.reward_list[self._progress_index()],
        )

    def _slot_arm(self, slot_id):
        return ArmTag("left" if self.cover_target_pose_xy[slot_id][0] < 0 else "right")

    def _visible_block_arm(self, block_id):
        return ArmTag(
            "left" if self.visible_blocks[block_id].get_pose().p[0] < 0 else "right"
        )

    def _target_cover_pose(self, slot_id, reveal=False):
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

        cover_xy = np.asarray(
            self.covers[slot_id].get_pose().p[:2],
            dtype=np.float32,
        )
        block_xy = np.asarray(
            self.covered_blocks[slot_id].get_pose().p[:2],
            dtype=np.float32,
        )

        dist_xy = float(np.linalg.norm(cover_xy - block_xy))

        if dist_xy > self._tracked_best_keyframe_dist + 1e-8:
            self._tracked_best_keyframe_dist = dist_xy
            self._tracked_best_keyframe_step = int(frame_idx)

    def get_keyframe_oracle_info(self):
        segment_active = False
        segment_slot = None
        dist_xy = None

        expected_slot = (
            int(self.inspect_pointer)
            if 0 <= self.inspect_pointer < self.num_reference_slots
            else None
        )

        segment_candidates = []

        if self.phase == "inspect":
            slot_dist_records = []

            for slot_id in range(self.num_reference_slots):
                slot_status = self._cover_status(slot_id)

                if slot_status == "covered":
                    continue

                cover_xy = np.asarray(
                    self.covers[slot_id].get_pose().p[:2],
                    dtype=np.float32,
                )
                block_xy = np.asarray(
                    self.covered_blocks[slot_id].get_pose().p[:2],
                    dtype=np.float32,
                )

                slot_dist = float(np.linalg.norm(cover_xy - block_xy))
                slot_priority = (
                    1 if expected_slot is not None and slot_id == expected_slot else 0
                )

                slot_dist_records.append((slot_priority, slot_dist, slot_id))

            if len(slot_dist_records) > 0:
                slot_dist_records.sort(
                    key=lambda item: (item[0], item[1]),
                    reverse=True,
                )

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
            "task_name": "pick_the_unhidden_block",
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

        dist_to_cover = np.linalg.norm(
            cover_pose[:2] - np.array(self.cover_target_pose_xy[slot_id])
        )
        dist_to_reveal = np.linalg.norm(
            cover_pose[:2] - np.array(self.reveal_pose_xy[slot_id])
        )

        if dist_to_cover < self.cover_dist_threshold and cover_pose[2] < self.cover_z_threshold:
            return "covered"

        if dist_to_reveal < self.reveal_dist_threshold:
            return "revealed"

        return "other"

    def _is_cover_back_xy(self, slot_id):
        cover_pose = self.covers[slot_id].get_pose().p

        dist_to_cover = np.linalg.norm(
            cover_pose[:2] - np.array(self.cover_target_pose_xy[slot_id])
        )

        return dist_to_cover < self.cover_dist_threshold * 1.5

    def _is_slot_open_for_progress(self, slot_id):
        status = self._cover_status(slot_id)

        if status == "revealed":
            return True

        cover_xy = np.asarray(
            self.covers[slot_id].get_pose().p[:2],
            dtype=np.float32,
        )
        reveal_xy = np.asarray(
            self.reveal_pose_xy[slot_id],
            dtype=np.float32,
        )

        dist_to_reveal = float(np.linalg.norm(cover_xy - reveal_xy))

        return dist_to_reveal < (self.reveal_dist_threshold * 1.25)

    def _all_covers_closed(self):
        # 不再严格要求 z < 0.742。
        # 对数据采集来说，只要 cover 的 XY 回到对应 block 上方即可认为关闭。
        # 严格 z 判断容易因为物理抖动导致 false negative。
        return all(
            self._is_cover_back_xy(slot_id)
            for slot_id in range(self.num_reference_slots)
        )

    def _visible_block_lifted(self, block_id):
        current_z = float(self.visible_blocks[block_id].get_pose().p[2])
        init_z = float(self.visible_block_init_z[block_id])

        return (
            current_z - init_z
            >= self.pickup_success_lift_height - self.pickup_success_tolerance
        )

    def _mark_inspection_complete(self, slot_id):
        if self.fail_flag:
            return

        if self.phase != "inspect" or slot_id != self.inspect_pointer:
            self.fail_flag = True
            return

        self.inspect_pointer += 1
        self.inspect_reveal_slot = None

        if self.inspect_pointer == self.num_reference_slots:
            self.phase = "pickup"

        self._update_reward()

    def update_state_transition(self):
        if self.fail_flag:
            return

        if self.phase == "inspect":
            if self.inspect_pointer >= self.num_reference_slots:
                self.phase = "pickup"
            else:
                expected_slot = self.inspect_pointer
                is_expected_open = self._is_slot_open_for_progress(expected_slot)

                # 与 cover_blocks_hard 对齐：
                # 不在这里额外检查其他 cover 是否轻微移动。
                # 这个额外检查在当前任务中容易因为接触扰动导致 fail。
                if self.inspect_reveal_slot is None:
                    if is_expected_open:
                        self.inspect_reveal_slot = expected_slot
                else:
                    if not is_expected_open:
                        self._mark_inspection_complete(expected_slot)

            return

        if self.phase == "pickup":
            if not self._all_covers_closed():
                return

            lifted_other_visible_block = any(
                self._visible_block_lifted(block_id)
                for block_id in range(self.num_visible_blocks)
                if block_id != self.target_visible_block_id
            )

            if lifted_other_visible_block:
                self.fail_flag = True
                return

            if self._visible_block_lifted(self.target_visible_block_id):
                self.pickup_done = True
                self.phase = "success"
                self._update_reward()

    def inspect_cover(self, slot_id):
        arm_tag = self._slot_arm(slot_id)

        instruction = (
            f"Open the {self.cover_name[slot_id]} cover to inspect the block color, "
            f"then place the cover back."
        )

        self._start_keyframe_tracking(slot_id)
        keyframe_step = None

        try:
            self.move(
                self.grasp_actor(
                    self.covers[slot_id],
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.05,
                ),
                language_annotation=instruction,
            )

            self.move(
                self.move_by_displacement(
                    arm_tag=arm_tag,
                    z=self.inspect_lift_height,
                ),
                language_annotation=instruction,
            )

            self.move(
                self.move_by_displacement(
                    arm_tag=arm_tag,
                    y=self.reveal_y_offset,
                ),
                language_annotation=instruction,
            )

            self.delay(
                delay_time=self.inspect_delay,
                language_annotation=instruction,
            )

            self.move(
                self.move_by_displacement(
                    arm_tag=arm_tag,
                    y=-self.reveal_y_offset,
                ),
                language_annotation=instruction,
            )

            self.move(
                self.place_actor(
                    self.covers[slot_id],
                    target_pose=self._target_cover_pose(slot_id, reveal=False),
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.05,
                    dis=0.005,
                ),
                language_annotation=instruction,
            )

            self.move(
                self.move_by_displacement(
                    arm_tag=arm_tag,
                    z=0.05,
                ),
                language_annotation=instruction,
            )

            self.move(
                self.back_to_origin(arm_tag=arm_tag),
                language_annotation=instruction,
            )

        finally:
            keyframe_step = self._finish_keyframe_tracking(slot_id)

        if self.plan_success:
            if keyframe_step is not None:
                self.keyframe_steps.append(keyframe_step)

            # 如果 check_success / update_state_transition 已经推进过，
            # 这里避免重复 mark 造成 fail_flag。
            if self.phase == "inspect" and slot_id == self.inspect_pointer:
                self._mark_inspection_complete(slot_id)

    def pick_target_block(self):
        arm_tag = self._visible_block_arm(self.target_visible_block_id)
        target_color_name = self.color_names[self.target_color_id]

        instruction = (
            f"Pick up the {target_color_name} block that did not appear under the covers."
        )

        self.move(
            self.grasp_actor(
                self.visible_blocks[self.target_visible_block_id],
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.02,
            ),
            language_annotation=instruction,
        )

        # demo 实际抬 10cm，保证稳定；
        # 成功判定只要求抬起约 1cm。
        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=self.pickup_demo_lift_height,
            ),
            language_annotation=instruction,
        )

        if self.plan_success:
            self.update_state_transition()

    def play_once(self):
        self.keyframe_steps = []

        for slot_id in range(self.num_reference_slots):
            self.inspect_cover(slot_id)

            if not self.plan_success or self.fail_flag:
                break

        if self.plan_success and not self.fail_flag:
            self.pick_target_block()

        self.info["info"] = {
            "keyframe_steps": list(self.keyframe_steps),
            "covered_color_order": [
                self.color_names[color_id]
                for color_id in self.covered_block_color_ids
            ],
            "visible_color_order": [
                self.color_names[color_id]
                for color_id in self.visible_block_color_ids
            ],
            "target_color": self.color_names[self.target_color_id],
            "target_visible_block_id": int(self.target_visible_block_id),
        }

        return self.info

    def check_success(self):
        self.update_state_transition()

        if self.fail_flag:
            return False

        if self.phase == "success" and self.pickup_done and self._all_covers_closed():
            self.max_reward = max(self.max_reward, 1.0)
            return True

        return False
