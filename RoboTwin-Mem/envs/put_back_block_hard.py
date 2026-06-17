from ._base_task import Base_Task
from .utils import *


class put_back_block_hard(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.button = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.25, -0.25],
            ylim=[-0.1, -0.1],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button.set_mass(0.0001, ["button_cap"])
        self.set_button_unpressed(self.button)
        self.press_cnt = 0
        self.press_flag = False

        self.block_half_size = 0.02
        self.center_threshold = 0.035
        self.pad_threshold = 0.03
        self.block_z_threshold = 0.77

        self.row_names = ["first", "second"]
        self.block_names = ["red", "green"]
        self.block_colors = [(1, 0, 0), (0, 1, 0)]
        self.side_names = ["left", "right"]
        self.row_y = [-0.03, -0.17]
        self.left_x = 0.02
        self.center_x = 0.14
        self.right_x = 0.26
        self.pad_half_size = [0.04, 0.04, 0.0005]
        self.center_marker_half_size = [0.035, 0.035, 0.0005]

        def create_block(block_pose, color, name):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=color,
                name=f"box_{name}",
            )

        def create_pad(pose, color, name):
            return create_box(
                scene=self,
                pose=pose,
                half_size=self.pad_half_size,
                color=color,
                name=name,
                is_static=True,
            )

        self.center_poses = []
        self.center_xys = []
        self.center_markers = []
        self.outer_pad_poses = []

        for row_id, y_pos in enumerate(self.row_y):
            center_pose = rand_pose(
                xlim=[self.center_x, self.center_x],
                ylim=[y_pos, y_pos],
                zlim=[0.7415],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            self.center_markers.append(
                create_box(
                    scene=self,
                    pose=center_pose,
                    half_size=self.center_marker_half_size,
                    color=(0.85, 0.85, 0.85),
                    name=f"center_marker_{self.row_names[row_id]}",
                    is_static=True,
                )
            )
            self.center_poses.append([self.center_x, y_pos, 0.765, 1, 0, 0, 0])
            self.center_xys.append(np.array([self.center_x, y_pos]))

            row_pad_poses = []
            for side_id, x_pos in enumerate([self.left_x, self.right_x]):
                pad_pose = rand_pose(
                    xlim=[x_pos, x_pos],
                    ylim=[y_pos, y_pos],
                    zlim=[0.7415],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=False,
                )
                create_pad(
                    pad_pose,
                    color=(0.000, 0.502, 0.996),
                    name=f"outer_pad_{self.row_names[row_id]}_{self.side_names[side_id]}",
                )
                row_pad_poses.append(np.array([x_pos, y_pos, 0.765]))
            self.outer_pad_poses.append(row_pad_poses)

        self.blocks = []
        for row_id, y_pos in enumerate(self.row_y):
            block_pose = rand_pose(
                xlim=[self.center_x, self.center_x],
                ylim=[y_pos, y_pos],
                zlim=[0.741 + self.block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            self.blocks.append(
                create_block(block_pose, self.block_colors[row_id], self.block_names[row_id])
            )

        self.demo_first_visit_side_ids = [int(np.random.randint(0, 2)), int(np.random.randint(0, 2))]
        self.actual_first_visit_side_ids = [None, None]
        self.phase = "in_progress"
        self.fail_flag = False
        self.reward_list = [0.0, 0.12, 0.24, 0.36, 0.5, 0.64, 0.76, 0.86, 0.94, 1.0]
        self.progress_index = 0
        self.keyframe_steps = []
        self.first_center_return_achieved = False
        self.second_center_return_achieved = False

    def _update_reward(self, reward_idx):
        self.progress_index = max(self.progress_index, reward_idx)
        self.max_reward = max(self.max_reward, self.reward_list[self.progress_index])

    def _append_keyframe_step(self):
        if not self.save_data or self.FRAME_IDX <= 0:
            return
        frame_idx = int(self.FRAME_IDX - 1)
        if len(self.keyframe_steps) == 0 or self.keyframe_steps[-1] != frame_idx:
            self.keyframe_steps.append(frame_idx)

    def _block_on_center(self, row_id):
        block_pose = self.blocks[row_id].get_pose().p
        return (
            np.linalg.norm(block_pose[:2] - self.center_xys[row_id]) < self.center_threshold
            and block_pose[2] < self.block_z_threshold
        )

    def _block_on_outer_pad(self, row_id, side_id):
        block_pose = self.blocks[row_id].get_pose().p
        target_pose = self.outer_pad_poses[row_id][side_id]
        return (
            np.abs(block_pose[0] - target_pose[0]) < self.pad_threshold
            and np.abs(block_pose[1] - target_pose[1]) < self.pad_threshold
            and block_pose[2] < self.block_z_threshold
        )

    def _block_on_any_outer_pad(self, row_id):
        for side_id in range(2):
            if self._block_on_outer_pad(row_id, side_id):
                return side_id
        return None

    def _block_in_wrong_row(self, row_id):
        block_pose = self.blocks[row_id].get_pose().p
        other_row_id = 1 - row_id
        if (
            np.linalg.norm(block_pose[:2] - self.center_xys[other_row_id]) < self.center_threshold
            and block_pose[2] < self.block_z_threshold
        ):
            return True
        for side_id in range(2):
            other_pose = self.outer_pad_poses[other_row_id][side_id]
            if (
                np.abs(block_pose[0] - other_pose[0]) < self.pad_threshold
                and np.abs(block_pose[1] - other_pose[1]) < self.pad_threshold
                and block_pose[2] < self.block_z_threshold
            ):
                return True
        return False

    def _validate_row_constraint(self):
        return not self._block_in_wrong_row(0) and not self._block_in_wrong_row(1)

    def _blocks_restored_to_selected_outer_pads(self):
        return all(
            self.actual_first_visit_side_ids[row_id] is not None
            and self._block_on_outer_pad(row_id, self.actual_first_visit_side_ids[row_id])
            for row_id in range(2)
        )

    def update_state_transition(self):
        first_center = self._block_on_center(0)
        second_center = self._block_on_center(1)
        first_outer = self._block_on_any_outer_pad(0)
        second_outer = self._block_on_any_outer_pad(1)
        if self.actual_first_visit_side_ids[0] is None and first_outer is not None:
            self.actual_first_visit_side_ids[0] = first_outer
            self._append_keyframe_step()
            self._update_reward(1)

        if self.actual_first_visit_side_ids[0] is not None and first_center:
            self.first_center_return_achieved = True
            self._update_reward(2)

        if self.actual_first_visit_side_ids[1] is None and second_outer is not None:
            self.actual_first_visit_side_ids[1] = second_outer
            self._append_keyframe_step()
            self._update_reward(4)

        if self.actual_first_visit_side_ids[1] is not None and second_center:
            self.second_center_return_achieved = True
            self._update_reward(5)

        if self.actual_first_visit_side_ids[0] is not None and self._block_on_outer_pad(0, self.actual_first_visit_side_ids[0]):
            self._update_reward(7)

        if self._blocks_restored_to_selected_outer_pads():
            self.phase = "wait_final_press"
            self._update_reward(8)

    def move_block_to_center(self, row_id, language_annotation):
        self.move(
            self.grasp_actor(self.blocks[row_id], arm_tag="right", pre_grasp_dis=0.08, grasp_dis=0.02),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=language_annotation)
        self.move(
            self.place_actor(
                self.blocks[row_id],
                arm_tag="right",
                target_pose=self.center_poses[row_id],
                functional_point_id=2,
                dis=0.015,
            ),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="right", z=0.08), language_annotation=language_annotation)
        self.move(self.back_to_origin(arm_tag="right"), language_annotation=language_annotation)
        if self.plan_success:
            self.update_state_transition()

    def move_block_to_outer_pad(self, row_id, side_id, language_annotation):
        self.move(
            self.grasp_actor(self.blocks[row_id], arm_tag="right", pre_grasp_dis=0.08, grasp_dis=0.02),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=language_annotation)
        self.move(
            self.place_actor(
                self.blocks[row_id],
                arm_tag="right",
                target_pose=self.outer_pad_poses[row_id][side_id],
                functional_point_id=2,
                dis=0.015,
            ),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="right", z=0.08), language_annotation=language_annotation)
        self.move(self.back_to_origin(arm_tag="right"), language_annotation=language_annotation)
        if self.plan_success:
            self.update_state_transition()

    def press_button(self, language_annotation):
        self.move(
            self.grasp_actor(self.button, arm_tag="left", pre_grasp_dis=0.08, grasp_dis=0.08, contact_point_id=0),
            self.back_to_origin(arm_tag="right"),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation=language_annotation)
        self.update_press_success()
        self.check_success()
        self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation=language_annotation)
        self.move(self.back_to_origin(arm_tag="left"), language_annotation=language_annotation)
        self.set_button_unpressed(self.button)
        self.update_button_reset(self.button)

    def play_once(self):
        self.keyframe_steps = []
        self.actual_first_visit_side_ids = [None, None]
        first_side_name = self.side_names[self.demo_first_visit_side_ids[0]]
        second_side_name = self.side_names[self.demo_first_visit_side_ids[1]]

        self.move_block_to_outer_pad(
            0,
            self.demo_first_visit_side_ids[0],
            language_annotation=f"Move the first-row {self.block_names[0]} block from the center to the {first_side_name} outer pad in the same row.",
        )
        if self.plan_success and not self.fail_flag:
            self.move_block_to_center(
                0,
                language_annotation="Move the first-row block back to the center position in its own row.",
            )
        if self.plan_success and not self.fail_flag:
            self.press_button("Press the button after finishing the first-row move-return sequence.")

        if self.plan_success and not self.fail_flag:
            self.move_block_to_outer_pad(
                1,
                self.demo_first_visit_side_ids[1],
                language_annotation=f"Move the second-row {self.block_names[1]} block from the center to the {second_side_name} outer pad in the same row.",
            )
        if self.plan_success and not self.fail_flag:
            self.move_block_to_center(
                1,
                language_annotation="Move the second-row block back to the center position in its own row.",
            )
        if self.plan_success and not self.fail_flag:
            self.press_button("Press the button after finishing the second-row move-return sequence.")

        if self.plan_success and not self.fail_flag:
            self.move_block_to_outer_pad(
                0,
                self.demo_first_visit_side_ids[0],
                language_annotation="Put the first-row block back onto the same outer pad it visited the first time.",
            )
        if self.plan_success and not self.fail_flag:
            self.move_block_to_outer_pad(
                1,
                self.demo_first_visit_side_ids[1],
                language_annotation="Put the second-row block back onto the same outer pad it visited the first time.",
            )
        if self.plan_success and not self.fail_flag:
            self.press_button("Press the button after both blocks are restored to their previously visited outer pads.")

        self.info["info"] = {
            "keyframe_steps": list(self.keyframe_steps),
            "first_visit_outer_pads": [
                None if side_id is None else self.side_names[side_id]
                for side_id in self.actual_first_visit_side_ids
            ],
        }
        return self.info

    def get_current_button_value(self, button_name, joint_name="button_joint", target=0.0):
        if button_name == "button":
            button_actor = self.button
        else:
            button_actor = self.check_button
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [j.get_name() for j in joints]
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        return qpos[idx]

    def set_button_unpressed(self, button_actor, joint_name="button_joint", target=0.0):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [j.get_name() for j in joints]
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        qpos[idx] = target
        art.set_qpos(qpos)
        joints[idx].set_drive_target(target)

    def check_button_pressed(self, button_actor, joint_name="button_joint", threshold=-0.005):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [j.get_name() for j in joints]
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        return qpos[idx] < threshold

    def update_button_reset(self, button_actor, joint_name="button_joint", threshold=-0.001):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [j.get_name() for j in joints]
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        if qpos[idx] > threshold:
            self.press_flag = False

    def update_press_success(self):
        if self.check_button_pressed(self.button) and not self.press_flag:
            self.press_flag = True
            self.press_cnt += 1

    def check_success(self):
        if self.phase == "success":
            self.max_reward = max(self.max_reward, 1.0)
            return True

        button_pressed = self.check_button_pressed(self.button)
        self.update_button_reset(self.button)
        self.update_press_success()
        self.set_button_unpressed(self.button, target=min(0.0, self.get_current_button_value("button") + 0.002))
        self.update_state_transition()

        if button_pressed and self._blocks_restored_to_selected_outer_pads():
            self.phase = "success"
            self._update_reward(9)
            self.max_reward = max(self.max_reward, 1.0)
            return True

        if self.phase == "success":
            self.max_reward = max(self.max_reward, 1.0)
            return True
        return False
