from ._base_task import Base_Task
from .utils import *


class reproduce_route(Base_Task):

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
        mat_half_size = [0.04, 0.04, 0.0005]

        self.center_pose = [0.1, -0.1, 0.765, 1, 0, 0, 0]

        self.mat_names = ["left", "right", "front", "back"]

        def create_block(block_pose, name):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(
                    self.block_half_size,
                    self.block_half_size,
                    self.block_half_size,
                ),
                color=(1, 0, 0),
                name=name,
            )

        def create_mat(mat_pose, name):
            return create_box(
                scene=self,
                pose=mat_pose,
                half_size=mat_half_size,
                color=(0.000, 0.502, 0.996),
                name=name,
                is_static=True,
            )

        mats_pose = []

        x_mat = 0.0
        for _ in range(2):
            mat_pos = rand_pose(
                xlim=[x_mat, x_mat],
                ylim=[-0.1, -0.1],
                qpos=[1, 0, 0, 0],
            )
            mats_pose.append(mat_pos)
            x_mat += 0.2

        y_mat = -0.2
        for _ in range(2):
            mat_pos = rand_pose(
                xlim=[0.1, 0.1],
                ylim=[y_mat, y_mat],
                qpos=[1, 0, 0, 0],
            )
            mats_pose.append(mat_pos)
            y_mat += 0.2

        self.mat_lst = []
        for i in range(4):
            mat = create_mat(mats_pose[i], name=f"mat_{self.mat_names[i]}")
            self.mat_lst.append(mat)

        block1_init_pose = rand_pose(
            xlim=[0.1, 0.1],
            ylim=[-0.1, -0.1],
            qpos=[1, 0, 0, 0],
        )

        block2_init_pose = rand_pose(
            xlim=[0.1, 0.1],
            ylim=[-0.28, -0.28],
            qpos=[1, 0, 0, 0],
        )

        self.block1 = create_block(block1_init_pose, name="red_block_1")
        self.block2 = create_block(block2_init_pose, name="red_block_2")

        self.route_ids = np.random.permutation(4).tolist()
        self.route_names = [self.mat_names[i] for i in self.route_ids]
        self.route_poses = [self.mat_lst[i].get_pose().p for i in self.route_ids]

        self.block1_route_progress = 0
        self.block2_route_progress = 0
        self.actual_route_ids = []

        # stage_id:
        # 0: block1 demonstrates the random route
        # 1: block1 has finished the route and should be placed back to center,
        #    then the first button press is required
        # 2: first button press done, block2 should reproduce the route
        # 3: block2 has finished reproducing the route, waiting for second button press
        # 4: success
        self.stage_id = 0
        self.stage1_start_press_cnt = 0
        self.stage3_start_press_cnt = 0

        self.keyframe_steps = []

    def _append_keyframe_step(self):
        if not self.save_data or self.FRAME_IDX <= 0:
            return

        frame_idx = int(self.FRAME_IDX - 1)
        if len(self.keyframe_steps) == 0 or self.keyframe_steps[-1] != frame_idx:
            self.keyframe_steps.append(frame_idx)

    def play_once(self):
        self.keyframe_steps = []
        # ----------------------------------------------------
        # Phase 1: use block1 to demonstrate a random route
        # ----------------------------------------------------
        for step_id, mat_id in enumerate(self.route_ids):
            mat_name = self.mat_names[mat_id]
            target_pose = self.route_poses[step_id]

            language = f"Move red block 1 to the {mat_name} mat."

            self.move(
                self.grasp_actor(
                    self.block1,
                    arm_tag="right",
                    pre_grasp_dis=0.08,
                    grasp_dis=0.02,
                ),
                language_annotation=language,
            )
            self.move(
                self.move_by_displacement(arm_tag="right", z=0.1),
                language_annotation=language,
            )
            self.move(
                self.place_actor(
                    self.block1,
                    arm_tag="right",
                    target_pose=target_pose,
                    functional_point_id=2,
                    dis=0.01,
                ),
                language_annotation=language,
            )

            if self.plan_success:
                self._append_keyframe_step()

            self.update_task_state()

        # ----------------------------------------------------
        # Phase 2: move block1 back to the center
        # ----------------------------------------------------
        language = "Move red block 1 back to the center position."

        self.move(
            self.grasp_actor(
                self.block1,
                arm_tag="right",
                pre_grasp_dis=0.08,
                grasp_dis=0.02,
            ),
            language_annotation=language,
        )
        self.move(
            self.move_by_displacement(arm_tag="right", z=0.1),
            language_annotation=language,
        )
        self.move(
            self.place_actor(
                self.block1,
                arm_tag="right",
                target_pose=self.center_pose,
                functional_point_id=2,
                dis=0.02,
            ),
            language_annotation=language,
        )

        self.update_task_state()

        self.move(
            self.back_to_origin(arm_tag="right"),
            language_annotation=language,
        )
        # ----------------------------------------------------
        # Phase 3: left arm presses the button
        # ----------------------------------------------------
        self.press_button()
        self.move(
            self.back_to_origin(arm_tag="left"),
            language_annotation="Press the button.",
        )
        # ----------------------------------------------------
        # Phase 4: block2 reproduces the same route
        # ----------------------------------------------------
        for step_id, mat_id in enumerate(self.route_ids):
            mat_name = self.mat_names[mat_id]
            target_pose = self.route_poses[step_id]

            language = f"Move red block 2 to the {mat_name} mat."

            self.move(
                self.grasp_actor(
                    self.block2,
                    arm_tag="right",
                    pre_grasp_dis=0.08,
                    grasp_dis=0.02,
                ),
                language_annotation=language,
            )
            self.move(
                self.move_by_displacement(arm_tag="right", z=0.1),
                language_annotation=language,
            )
            self.move(
                self.place_actor(
                    self.block2,
                    arm_tag="right",
                    target_pose=target_pose,
                    functional_point_id=2,
                    dis=0.01,
                ),
                language_annotation=language,
            )

            self.update_task_state()

        self.move(
            self.back_to_origin(arm_tag="right"),
            language_annotation=language,
        )
        # ----------------------------------------------------
        # Phase 5: press button again after block2 finishes
        # ----------------------------------------------------
        self.press_button()
        self.move(
            self.back_to_origin(arm_tag="left"),
            language_annotation="Press the button.",
        )

        self.info["info"] = {
            "keyframe_steps": list(self.keyframe_steps),
        }
        return self.info

    def press_button(self):
        self.move(
            self.grasp_actor(
                self.button,
                arm_tag="left",
                pre_grasp_dis=0.08,
                grasp_dis=0.08,
                contact_point_id=0,
            ),
            language_annotation="Press the button.",
        )
        self.move(
            self.move_by_displacement(arm_tag="left", z=-0.04),
            language_annotation="Press the button.",
        )

        self.update_task_state()

        self.move(
            self.move_by_displacement(arm_tag="left", z=0.04),
            language_annotation="Press the button.",
        )
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

    def update_press_success(self):
        if self.check_button_pressed(self.button) and not self.press_flag:
            self.press_flag = True
            self.press_cnt += 1

    def check_actor_at_pose(self, actor, target_pose, xy_threshold=0.035, z_threshold=0.77):
        actor_pose = actor.get_pose().p

        return (
            np.abs(actor_pose[0] - target_pose[0]) < xy_threshold
            and np.abs(actor_pose[1] - target_pose[1]) < xy_threshold
            and actor_pose[2] < z_threshold
        )

    def check_actor_on_any_mat(self, actor):
        for mat_id, mat_pose in enumerate(self.route_poses):
            if self.check_actor_at_pose(actor, mat_pose, xy_threshold=0.035):
                return mat_id
        return None

    def check_block1_in_center(self):
        block_pose = self.block1.get_pose().p

        return (
            np.abs(block_pose[0] - self.center_pose[0]) < 0.04
            and np.abs(block_pose[1] - self.center_pose[1]) < 0.04
            and block_pose[2] < 0.77
        )

    def update_block1_route_progress(self):
        if len(self.actual_route_ids) >= len(self.route_poses):
            return

        if not self.is_right_gripper_open():
            return

        mat_id = self.check_actor_on_any_mat(self.block1)
        if mat_id is None:
            return

        if mat_id not in self.actual_route_ids:
            self.actual_route_ids.append(mat_id)
            self.block1_route_progress = len(self.actual_route_ids)

    def update_block2_route_progress(self):
        if self.block2_route_progress >= len(self.actual_route_ids):
            return

        target_mat_id = self.actual_route_ids[self.block2_route_progress]
        target_pose = self.route_poses[target_mat_id]

        if self.is_right_gripper_open() and self.check_actor_at_pose(
            self.block2,
            target_pose,
            xy_threshold=0.035,
        ):
            self.block2_route_progress += 1

    def update_task_state(self):
        self.update_button_reset(self.button)
        self.update_press_success()

        self.set_button_unpressed(
            self.button,
            target=min(0.0, self.get_current_button_value("button") + 0.002),
        )

        # ----------------------------------------------------
        # Stage 0: block1 follows the random route
        # ----------------------------------------------------
        if self.stage_id == 0:
            self.update_block1_route_progress()

            if len(self.actual_route_ids) >= len(self.route_poses):
                self.stage_id = 1

                # 进入 stage 1 时记录当前按钮次数
                # 这样可以避免 stage 0 期间提前按按钮被算作第一次按钮
                self.stage1_start_press_cnt = self.press_cnt

            return

        # ----------------------------------------------------
        # Stage 1: block1 should be placed back to center,
        # then a new button press is required
        # ----------------------------------------------------
        if self.stage_id == 1:
            block1_center = self.check_block1_in_center()

            # 如果 block1 还没回到 center，则持续刷新基准 press_cnt
            # 这样可以保证按钮必须发生在 block1 回 center 之后
            if not block1_center:
                self.stage1_start_press_cnt = self.press_cnt
                return

            if self.press_cnt > self.stage1_start_press_cnt:
                self.stage_id = 2

            return

        # ----------------------------------------------------
        # Stage 2: block2 reproduces the same route
        # ----------------------------------------------------
        if self.stage_id == 2:
            self.update_block2_route_progress()

            if self.block2_route_progress >= len(self.route_poses):
                self.stage_id = 3

                # 进入 stage 3 时记录当前按钮次数
                # 这样可以避免第一次按钮被算作最终按钮
                self.stage3_start_press_cnt = self.press_cnt

            return

        # ----------------------------------------------------
        # Stage 3: after block2 finishes the route, a new button press completes the task
        # ----------------------------------------------------
        if self.stage_id == 3:
            if self.press_cnt > self.stage3_start_press_cnt:
                self.stage_id = 4
                self.max_reward = max(self.max_reward, 1.0)

            return

        # Stage 4: already success
        if self.stage_id == 4:
            self.max_reward = max(self.max_reward, 1.0)
            return

    def check_success(self):
        self.update_task_state()

        if self.stage_id == 4:
            self.max_reward = max(self.max_reward, 1.0)
            return True

        return False
