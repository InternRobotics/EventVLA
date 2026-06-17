from ._base_task import Base_Task
from .utils import *


class rearrange_blocks_hard(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.button = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.25, -0.15],
            ylim=[-0.2, -0.1],
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
        self.block_y = np.random.uniform(-0.15, -0.08)
        self.center_pose = [0.14, self.block_y, 0.765, 1, 0, 0, 0]
        self.center_xy = np.array(self.center_pose[:2])
        self.center_threshold = 0.03
        self.mat_threshold = 0.03
        self.block_z_threshold = 0.77

        mat_half_size = [0.04, 0.04, 0.0005]
        self.mat_names = ["left", "right"]
        self.block_names = ["red", "green"]
        self.block_colors = [(1, 0, 0), (0, 1, 0)]
        self.mat_x = [0.02, 0.26]

        def create_block(block_pose, color, color_name):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=color,
                name=f"box_{color_name}",
            )

        def create_mat(mat_pose):
            return create_box(
                scene=self,
                pose=mat_pose,
                half_size=mat_half_size,
                color=(0.000, 0.502, 0.996),
                name="box",
                is_static=True,
            )

        mats_pose = []
        block_pose_lst = []
        for x_mat in self.mat_x:
            mat_pose = rand_pose(
                xlim=[x_mat, x_mat],
                ylim=[self.block_y, self.block_y],
                zlim=[0.7415],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            mats_pose.append(mat_pose)

            block_pose = rand_pose(
                xlim=[x_mat, x_mat],
                ylim=[self.block_y, self.block_y],
                zlim=[0.741 + self.block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            block_pose_lst.append(block_pose)

        self.mats = [create_mat(mat_pose) for mat_pose in mats_pose]
        self.blocks = [
            create_block(block_pose_lst[i], self.block_colors[i], self.block_names[i])
            for i in range(2)
        ]
        self.mat_target_poses = [self.mats[i].get_pose().p for i in range(2)]

        self.demo_first_block_id = int(np.random.randint(0, 2))
        self.demo_other_block_id = 1 - self.demo_first_block_id
        self.actual_first_block_id = None
        self.actual_other_block_id = None
        self.layout_restored_achieved = False
        self.other_center_achieved = False

        self.phase = "move_first_to_center"
        self.fail_flag = False
        self.reward_list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        self.progress_index = 0
        self.keyframe_steps = []

    def _update_reward(self, reward_idx):
        self.progress_index = max(self.progress_index, reward_idx)
        self.max_reward = max(self.max_reward, self.reward_list[self.progress_index])

    def _append_keyframe_step(self):
        if not self.save_data or self.FRAME_IDX <= 0:
            return
        frame_idx = int(self.FRAME_IDX - 1)
        if len(self.keyframe_steps) == 0 or self.keyframe_steps[-1] != frame_idx:
            self.keyframe_steps.append(frame_idx)

    def _block_on_center(self, block_id):
        block_pose = self.blocks[block_id].get_pose().p
        return (
            np.linalg.norm(block_pose[:2] - self.center_xy) < self.center_threshold
            and block_pose[2] < self.block_z_threshold
        )

    def _block_on_home(self, block_id):
        block_pose = self.blocks[block_id].get_pose().p
        target_pose = self.mat_target_poses[block_id]
        return (
            np.abs(block_pose[0] - target_pose[0]) < self.mat_threshold
            and np.abs(block_pose[1] - target_pose[1]) < self.mat_threshold
            and block_pose[2] < self.block_z_threshold
        )

    def _record_actual_first_block(self):
        if self.actual_first_block_id is not None or self.press_cnt > 1:
            return

        block0_on_center = self._block_on_center(0)
        block1_on_center = self._block_on_center(1)
        block0_on_home = self._block_on_home(0)
        block1_on_home = self._block_on_home(1)

        if block0_on_center and block1_on_home and not block1_on_center:
            self.actual_first_block_id = 0
            self.actual_other_block_id = 1
        elif block1_on_center and block0_on_home and not block0_on_center:
            self.actual_first_block_id = 1
            self.actual_other_block_id = 0

    def _record_achieved_progress(self, first_on_center, other_on_center, first_on_home, other_on_home):
        if self.actual_first_block_id is None:
            return

        all_home = first_on_home and other_on_home and not first_on_center and not other_on_center
        if self.press_cnt >= 1 and all_home:
            self.layout_restored_achieved = True

        # After the second button, only require that the other block truly reaches center.
        # The first block has already been confirmed to have been restored once.
        if self.press_cnt >= 2 and other_on_center and not first_on_center and self.layout_restored_achieved:
            self.other_center_achieved = True

    def update_state_transition(self):
        if self.fail_flag:
            return

        if self.press_cnt > 3:
            self.fail_flag = True
            return

        self._record_actual_first_block()
        if self.actual_first_block_id is None:
            if self.press_cnt > 1:
                self.fail_flag = True
            return

        first_on_center = self._block_on_center(self.actual_first_block_id)
        other_on_center = self._block_on_center(self.actual_other_block_id)
        first_on_home = self._block_on_home(self.actual_first_block_id)
        other_on_home = self._block_on_home(self.actual_other_block_id)
        all_home = first_on_home and other_on_home and not first_on_center and not other_on_center
        self._record_achieved_progress(first_on_center, other_on_center, first_on_home, other_on_home)

        if self.phase == "move_first_to_center":
            if self.press_cnt > 0 or other_on_center:
                self.fail_flag = True
                return
            if first_on_center and other_on_home:
                self._append_keyframe_step()
                self.phase = "wait_first_press"
                self._update_reward(1)
            return

        if self.phase == "wait_first_press":
            if self.press_cnt == 0:
                if not (first_on_center and other_on_home):
                    self.fail_flag = True
                return
            if self.press_cnt == 1 and first_on_center and other_on_home:
                self.phase = "restore_layout"
                self._update_reward(2)
                return
            self.fail_flag = True
            return

        if self.phase == "restore_layout":
            if self.press_cnt != 1 or other_on_center:
                self.fail_flag = True
                return
            if all_home:
                self.phase = "wait_second_press"
                self._update_reward(3)
            return

        if self.phase == "wait_second_press":
            if self.press_cnt == 1:
                if not self.layout_restored_achieved:
                    self.fail_flag = True
                return
            if self.press_cnt == 2 and self.layout_restored_achieved:
                self.phase = "move_other_to_center"
                self._update_reward(4)
                return
            self.fail_flag = True
            return

        if self.phase == "move_other_to_center":
            if self.press_cnt != 2 or first_on_center:
                self.fail_flag = True
                return
            if self.other_center_achieved:
                self.phase = "wait_third_press"
                self._update_reward(5)
            return

        if self.phase == "wait_third_press":
            if self.press_cnt == 2:
                if not self.other_center_achieved:
                    self.fail_flag = True
                return
            if self.press_cnt == 3 and self.other_center_achieved:
                self.phase = "success"
                self._update_reward(6)
                return
            self.fail_flag = True

    def move_block_to_center(self, block_id, language_annotation):
        self.move(
            self.grasp_actor(self.blocks[block_id], arm_tag="right", pre_grasp_dis=0.1, grasp_dis=0.02),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=language_annotation)
        self.move(
            self.place_actor(
                self.blocks[block_id],
                arm_tag="right",
                target_pose=self.center_pose,
                functional_point_id=2,
                dis=0.01,
            ),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="right", z=0.08), language_annotation=language_annotation)
        self.move(self.back_to_origin(arm_tag="right"), language_annotation=language_annotation)
        if self.plan_success:
            self.update_state_transition()

    def move_block_back_home(self, block_id, language_annotation):
        self.move(
            self.grasp_actor(self.blocks[block_id], arm_tag="right", pre_grasp_dis=0.05, grasp_dis=0.02),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=language_annotation)
        self.move(
            self.place_actor(
                self.blocks[block_id],
                arm_tag="right",
                target_pose=self.mat_target_poses[block_id],
                functional_point_id=2,
                dis=0.01,
            ),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="right", z=0.08), language_annotation=language_annotation)
        self.move(self.back_to_origin(arm_tag="right"), language_annotation=language_annotation)
        if self.plan_success:
            self.update_state_transition()

    def play_once(self):
        self.keyframe_steps = []
        first_name = self.block_names[self.demo_first_block_id]
        first_home = self.mat_names[self.demo_first_block_id]

        self.move_block_to_center(
            self.demo_first_block_id,
            language_annotation=f"Pick up the {first_name} block and move it to the center position.",
        )
        if self.plan_success and not self.fail_flag:
            self.press_button("Press the button.")
        if self.plan_success and not self.fail_flag:
            self.move_block_back_home(
                self.demo_first_block_id,
                language_annotation=f"Move the {first_name} block back to the {first_home} mat.",
            )
        if self.plan_success and not self.fail_flag:
            self.press_button("Press the button again after restoring the layout.")
        if self.plan_success and not self.fail_flag:
            self.move_block_to_center(
                self.demo_other_block_id,
                language_annotation="Now move the other block to the center position.",
            )
        if self.plan_success and not self.fail_flag:
            self.press_button("Press the button for the final time.")

        self.info["info"] = {"keyframe_steps": list(self.keyframe_steps)}
        return self.info

    def press_button(self, language_annotation):
        self.move(
            self.grasp_actor(self.button, arm_tag="left", pre_grasp_dis=0.08, grasp_dis=0.08, contact_point_id=0),
            self.back_to_origin(arm_tag="right"),
            language_annotation=language_annotation,
        )
        self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation=language_annotation)
        self.check_press_success()
        self.check_success()
        self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation=language_annotation)
        self.move(self.back_to_origin(arm_tag="left"), language_annotation=language_annotation)
        self.set_button_unpressed(self.button)
        self.update_button_reset(self.button)

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

    def check_press_success(self):
        if self.check_button_pressed(self.button) and not self.press_flag:
            self.press_flag = True
            self.press_cnt += 1

    def check_success(self):
        if self.fail_flag:
            return False
        if self.phase == "success":
            self.max_reward = max(self.max_reward, 1.0)
            return True

        self.update_button_reset(self.button)
        self.check_press_success()
        self.set_button_unpressed(self.button, target=min(0.0, self.get_current_button_value("button") + 0.002))
        self.update_state_transition()

        if self.fail_flag:
            return False
        if self.phase == "success":
            self.max_reward = max(self.max_reward, 1.0)
            return True
        return False
