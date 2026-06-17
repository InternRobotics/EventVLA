from ._base_task import Base_Task
from .utils import *
from copy import deepcopy
import numpy as np


class pick_objects_in_order(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.cover_num = 3
        self.cover_x = [-0.3, -0.1, 0.1]
        self.cover_y = [-0.12] * self.cover_num
        self.cover_names = ["left", "middle", "right"]
        self.quat_of_cover_target_pose = [0.0, 1.0, 0.0, 0.0]

        self.object_modelnames = [
            "009_toycar",
            "010_mouse",
            "011_stapler",
            "013_playingcards",
        ]

        self.object_names = [
            "toy car",
            "mouse",
            "stapler",
            "playing cards",
        ]

        # Randomly choose 3 unique objects under the 3 covers.
        self.hidden_object_ids = np.random.choice(
            len(self.object_modelnames),
            size=self.cover_num,
            replace=False,
        ).tolist()

        # The front objects are the same 3 objects, but randomly shuffled.
        self.front_object_ids = self.hidden_object_ids.copy()
        np.random.shuffle(self.front_object_ids)

        # For each hidden object from left to right,
        # find its corresponding index among the front objects.
        # Example:
        # hidden_object_ids = [mouse, stapler, toycar]
        # front_object_ids  = [toycar, mouse, stapler]
        # correct_front_pick_indices = [1, 2, 0]
        self.correct_front_pick_indices = [
            self.front_object_ids.index(obj_id)
            for obj_id in self.hidden_object_ids
        ]

        self.cover_pose_lst = []

        def create_cover(cover_pose):
            return create_actor(
                self,
                pose=cover_pose,
                modelname="003_cover",
                model_id=0,
                convex=True,
            )

        self.covers = []
        for i in range(self.cover_num):
            cover_pose = rand_pose(
                xlim=[self.cover_x[i], self.cover_x[i]],
                ylim=[self.cover_y[i], self.cover_y[i]],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )
            self.cover_pose_lst.append(deepcopy(cover_pose))
            self.covers.append(create_cover(cover_pose))

        self.hidden_objects = []
        for i, obj_id in enumerate(self.hidden_object_ids):
            obj_pose = rand_pose(
                xlim=[self.cover_x[i], self.cover_x[i]],
                ylim=[self.cover_y[i], self.cover_y[i]],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=False,
            )

            obj = create_actor(
                scene=self,
                pose=obj_pose,
                modelname=self.object_modelnames[obj_id],
                model_id=0,
                convex=True,
            )
            obj.set_mass(0.05)
            self.hidden_objects.append(obj)

        self.front_y = -0.28
        self.front_objects = []
        self.front_object_init_poses = []

        for i, obj_id in enumerate(self.front_object_ids):
            obj_pose = rand_pose(
                xlim=[self.cover_x[i], self.cover_x[i]],
                ylim=[self.front_y, self.front_y],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=False,
            )
            self.front_object_init_poses.append(deepcopy(obj_pose))

            obj = create_actor(
                scene=self,
                pose=obj_pose,
                modelname=self.object_modelnames[obj_id],
                model_id=0,
                convex=True,
            )
            obj.set_mass(0.05)
            self.front_objects.append(obj)

        self.cover_opened_once = [False] * self.cover_num
        self.prev_front_lifted_flags = [False] * self.cover_num
        self.front_object_lifted_once = [False] * self.cover_num
        self.picked_front_order = []
        self.pick_progress = 0
        self.fail_flag = False

        # stage_id:
        # 0: open covers and memorize objects
        # 1: pick front objects according to remembered left-to-right hidden order
        # 2: success
        self.stage_id = 0
        self.keyframe_steps = []

    def _append_keyframe_step(self):
        if not self.save_data or self.FRAME_IDX <= 0:
            return

        frame_idx = int(self.FRAME_IDX - 1)

        if len(self.keyframe_steps) == 0 or self.keyframe_steps[-1] != frame_idx:
            self.keyframe_steps.append(frame_idx)

    def _update_history_flags(self):
        open_height_thresh = 0.05

        for i in range(self.cover_num):
            cur_pose = self.covers[i].get_pose()
            init_pose = self.cover_pose_lst[i]

            if cur_pose is not None and init_pose is not None:
                if cur_pose.p[2] > init_pose.p[2] + open_height_thresh:
                    self.cover_opened_once[i] = True

    def _update_pick_order_from_current_state(self):
        if self.fail_flag:
            return

        pick_height_thresh = 0.08 

        current_lifted_flags = []
        for i in range(self.cover_num):
            cur_pose = self.front_objects[i].get_pose()
            init_pose = self.front_object_init_poses[i]

            if cur_pose is None or init_pose is None:
                current_lifted_flags.append(False)
                continue

            current_lifted_flags.append(
                cur_pose.p[2] > init_pose.p[2] + pick_height_thresh
            )

        newly_lifted_indices = []
        for i in range(self.cover_num):
            if current_lifted_flags[i] and not self.prev_front_lifted_flags[i]:
                newly_lifted_indices.append(i)

        self.prev_front_lifted_flags = current_lifted_flags

        if len(newly_lifted_indices) == 0:
            return

        newly_lifted_indices = [
            i for i in newly_lifted_indices
            if not self.front_object_lifted_once[i]
        ]

        if len(newly_lifted_indices) == 0:
            return

        if self.stage_id != 1:
            self.fail_flag = True
            self.fail_reason = "front object picked before all covers were opened"
            return

        if len(newly_lifted_indices) > 1:
            self.fail_flag = True
            self.fail_reason = f"multiple front objects lifted: {newly_lifted_indices}"
            return

        front_idx = newly_lifted_indices[0]

        if self.pick_progress >= self.cover_num:
            self.fail_flag = True
            self.fail_reason = "extra front object picked after task completion"
            return

        expected_front_idx = self.correct_front_pick_indices[self.pick_progress]

        if front_idx != expected_front_idx:
            self.fail_flag = True
            return

        self.front_object_lifted_once[front_idx] = True
        self.picked_front_order.append(front_idx)
        self.pick_progress += 1

    def _obj_name(self, obj_id):
        return self.object_names[obj_id]

    def _cover_original_target_pose(self, cover_idx):
        init_pose = self.cover_pose_lst[cover_idx]
        return [
            float(init_pose.p[0]),
            float(init_pose.p[1]),
            float(init_pose.p[2]),
        ] + self.quat_of_cover_target_pose

    def open_cover_and_put_back(self, cover_idx):
        name = self.cover_names[cover_idx]
        hidden_obj_id = self.hidden_object_ids[cover_idx]
        hidden_obj_name = self._obj_name(hidden_obj_id)

        cover_pose = self.covers[cover_idx].get_pose().p
        x = cover_pose[0]

        arm_tag = ArmTag("left" if x < 0 else "right")

        observe_language = (
            f"Open the {name} cover and observe that the object under it is the {hidden_obj_name}."
        )

        self.move(
            self.grasp_actor(
                self.covers[cover_idx],
                arm_tag=arm_tag,
                pre_grasp_dis=0.05,
            ),
            language_annotation=observe_language,
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.08),
            language_annotation=observe_language,
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, y=0.11),
            language_annotation=observe_language,
        )

        if self.plan_success:
            self.cover_opened_once[cover_idx] = True
            self._append_keyframe_step()

        self.move(
            self.place_actor(
                self.covers[cover_idx],
                target_pose=self._cover_original_target_pose(cover_idx),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.005,
            ),
            language_annotation=observe_language,
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.1),
            language_annotation=observe_language,
        )
        self.move(
            self.back_to_origin(arm_tag=arm_tag),
            language_annotation=observe_language,
        )

    def _record_front_pick_event(self, front_idx):
        if self.front_object_lifted_once[front_idx]:
            return

        if self.stage_id == 0 and sum(self.cover_opened_once) >= self.cover_num:
            self.stage_id = 1

        if self.stage_id != 1:
            self.fail_flag = True
            self.fail_reason = "front object picked before all covers were opened"
            return

        expected_front_idx = self.correct_front_pick_indices[self.pick_progress]
        if front_idx != expected_front_idx:
            self.fail_flag = True
            return

        self.front_object_lifted_once[front_idx] = True
        self.picked_front_order.append(front_idx)
        self.pick_progress += 1

    def pick_front_object_in_memory_order(self, order_idx):
        front_idx = self.correct_front_pick_indices[order_idx]
        obj_id = self.front_object_ids[front_idx]
        obj_name = self._obj_name(obj_id)

        obj_pose = self.front_objects[front_idx].get_pose().p
        arm_tag = ArmTag("left" if obj_pose[0] < 0 else "right")

        cover_name = self.cover_names[order_idx]
        language = (
            f"Pick up the {obj_name} which matches the object observed under the {cover_name} cover."
        )

        self.move(
            self.grasp_actor(
                self.front_objects[front_idx],
                arm_tag=arm_tag,
                pre_grasp_dis=0.06,
            ),
            language_annotation=language,
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.10),
            language_annotation=language,
        )
        if self.plan_success:
            self._record_front_pick_event(front_idx)

        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=-0.09),
            language_annotation=language,
        )
        self.move(self.open_gripper(arm_tag=arm_tag), language_annotation=language)
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.08),
            language_annotation=language,
        )
        self.move(
            self.back_to_origin(arm_tag=arm_tag),
            language_annotation=language,
        )

    def play_once(self):
        # Phase 1: open all covers from left to right and memorize objects.
        for cover_idx in range(self.cover_num):
            self.open_cover_and_put_back(cover_idx)

        # Phase 2: pick front objects according to hidden left-to-right object order.
        for order_idx in range(self.cover_num):
            self.pick_front_object_in_memory_order(order_idx)

        self.info["info"] = {
            "keyframe_steps": list(self.keyframe_steps),
        }
        return self.info

    def check_success(self):
        if self.fail_flag:
            return False
        # 1. Always update cover-opening history.
        self._update_history_flags()
        opened_num = sum(self.cover_opened_once)

        # 2. After all covers have been opened once, enter pick stage.
        if self.stage_id == 0:
            if opened_num >= self.cover_num:
                self.stage_id = 1

        # 3. Always detect newly lifted front objects and check strict pick order.
        self._update_pick_order_from_current_state()
        if self.fail_flag:
            return False

        # 4. If all three picks are done in correct order, enter success stage.
        if self.stage_id == 1:
            if self.pick_progress >= self.cover_num:
                self.stage_id = 2

        progress = opened_num + self.pick_progress
        self.max_reward = max(self.max_reward, progress / (self.cover_num * 2))

        if self.stage_id == 2:
            self.max_reward = max(self.max_reward, 1.0)
            return (
                self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open()
            )

        return False
