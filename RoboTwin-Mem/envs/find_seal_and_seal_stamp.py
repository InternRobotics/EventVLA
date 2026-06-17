from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import time
import numpy as np


class find_seal_and_seal_stamp(Base_Task):
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.cover_pose_lst = []
        self.cover_x, self.cover_y = [-0.3, -0.1, 0.1, 0.3], [-0.14] * 4
        self.cover_name = ["left", "middle_left", "middle_right", "right"]
        for i in range(4):
            cover_pose = rand_pose(
                xlim=[self.cover_x[i], self.cover_x[i]],
                ylim=[self.cover_y[i], self.cover_y[i]],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )
            self.cover_pose_lst.append(deepcopy(cover_pose))
        self.quat_of_target_pose = [0.0, 1, 0.0, 0.0]
        
        def create_cover(cover_pose):
            return create_actor(
                self, 
                pose=cover_pose, 
                modelname="003_cover", 
                model_id=0, 
                convex=True
            )
        self.covers = []
        for cover_pose in self.cover_pose_lst:
            cover = create_cover(cover_pose)
            self.covers.append(cover)

        x_pos = np.random.choice([-0.3, -0.1, 0.1, 0.3])
        if x_pos == -0.3:
            self.seal_in_cover = [1, 0, 0, 0]
        elif x_pos == -0.1:
            self.seal_in_cover = [0, 1, 0, 0]
        elif x_pos == 0.1:
            self.seal_in_cover = [0, 0, 1, 0]
        else:
            self.seal_in_cover = [0, 0, 0, 1]
        rand_pos = rand_pose(
            xlim=[x_pos, x_pos],
            ylim=[-0.14, -0.14],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        # self.seal_id = np.random.choice([0, 2, 3, 4, 6], 1)[0]
        self.seal = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="100_seal",
            convex=True,
            model_id=6,
        )
        self.seal.set_mass(0.05)

        target_rand_pose = rand_pose(
            xlim=[-0.03, 0.03],
            ylim=[-0.29, -0.27],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
            "Orange": (1, 0.5, 0),
            "Purple": (0.5, 0, 0.5),
            "Brown": (0.65, 0.4, 0.16),
            "Pink": (1, 0.75, 0.8),
            "Lime": (0.5, 1, 0),
            "Olive": (0.5, 0.5, 0),
            "Teal": (0, 0.5, 0.5),
            "Maroon": (0.5, 0, 0),
            "Navy": (0, 0, 0.5),
            "Coral": (1, 0.5, 0.31),
            "Turquoise": (0.25, 0.88, 0.82),
            "Indigo": (0.29, 0, 0.51),
            "Beige": (0.96, 0.91, 0.81),
            "Tan": (0.82, 0.71, 0.55),
            "Silver": (0.75, 0.75, 0.75),
        }

        color_items = list(colors.items())
        idx = np.random.choice(len(color_items))
        self.color_name, self.color_value = color_items[idx]

        half_size = [0.035, 0.035, 0.0005]
        self.target = create_visual_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
        )
        self.add_prohibit_area(self.seal, padding=0.1)
        self.add_prohibit_area(self.target, padding=0.1)
        self.target_pose = self.target.get_pose()

        self.seal_in_stamp_once = False
        self.cover_opened_once = [False] * 4
        self.cond_open_required_covers = False
        
        self.keyframe_steps = []

    def _append_keyframe_step(self):
        if not self.save_data or self.FRAME_IDX <= 0:
            return
        frame_idx = int(self.FRAME_IDX - 1)
        if len(self.keyframe_steps) == 0 or self.keyframe_steps[-1] != frame_idx:
            self.keyframe_steps.append(frame_idx)

    def play_once(self):
        self.keyframe_steps = []
        self.last_gripper = None
        for i in range(4):
            arm_tag = ArmTag("left" if self.covers[i].get_pose().p[0]<0 else "right")
            if not arm_tag == self.last_gripper and not self.last_gripper == None:
                self.move(self.back_to_origin(arm_tag=self.last_gripper), language_annotation=self.last_annotation)
            name = ["left", "middle_left", "middle_right", "right"][i]
            cover_pose = self.covers[i].get_pose().p
            x, y = cover_pose[0], cover_pose[1]
            self.move(self.grasp_actor(self.covers[i], arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f"Pick up the {name} cover and look for seal.")
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08), language_annotation=f"Pick up the {name} cover and look for seal.")
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.11), language_annotation=f"Pick up the {name} cover and look for seal.")
            self._update_history_flags()

            if self.plan_success:
                self._append_keyframe_step()

            if not self.seal_in_cover[i] == 1:
                target_pose = [x, y, 0.741]
                self.move(self.place_actor(self.covers[i], target_pose=target_pose + self.quat_of_target_pose, arm_tag=arm_tag, functional_point_id=0, pre_dis=0.05, dis=0.005), language_annotation=f"Pick up the {name} cover and look for seal.")
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f"Pick up the {name} cover and look for seal.")

            else:
                target_pose = [x, y+0.13, 0.741]
                self.idx=i
                self.move(self.place_actor(self.covers[i], target_pose=target_pose + self.quat_of_target_pose, arm_tag=arm_tag, functional_point_id=0, pre_dis=0.05, dis=0.005), language_annotation=f"Pick up the {name} cover and look for seal.")
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f"Pick up the {name} cover and look for seal.")
                self.move(self.grasp_actor(self.seal, arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f"Pick up the {name} cover and look for seal.")
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08), language_annotation=f"Take out the seal in the {name} cover.")
                self.move(self.place_actor(self.seal, target_pose=[x, y-0.15, 0.741] + [1,0,0,0], arm_tag=arm_tag, pre_dis=0.05, dis=0.005), language_annotation=f"Take out the seal in the {name} cover.")
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f"Take out the seal in the {name} cover.")
                self.move(self.grasp_actor(self.covers[i], arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f"Put the {name} cover backto its original position.")
                self.move(self.place_actor(self.covers[i], target_pose=[x, y, 0.741] + self.quat_of_target_pose, arm_tag=arm_tag, functional_point_id=0, pre_dis=0.05, dis=0.005), language_annotation=f"Put the {name} cover back to its original position.")
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f"Put the {name} cover back to its original position.")
                break
            self.last_gripper = arm_tag
            self.last_annotation = f"Put the {name} cover back to its original position."

        self.move(self.grasp_actor(self.seal, arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f"Pick up the seal and stamp it.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f"Pick up the seal and stamp it.")
        self.move(self.place_actor(self.seal, target_pose=self.target_pose, arm_tag=arm_tag, pre_dis=0.05, dis=0.005), language_annotation=f"Pick up the seal and stamp it.")
        self.seal_in_stamp_once = True
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.grasp_actor(self.covers[self.idx], arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.place_actor(self.covers[self.idx], target_pose=target_pose + self.quat_of_target_pose, arm_tag=arm_tag, functional_point_id=0, pre_dis=0.05, dis=0.005), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self.move(self.grasp_actor(self.seal, arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.18), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.place_actor(self.seal, target_pose=[x, y, 0.741] + [1,0,0,0], arm_tag=arm_tag, pre_dis=0.05, dis=0.005), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.18), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.grasp_actor(self.covers[self.idx], arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f"Put the seal back into the original lid.")
        self.move(self.place_actor(self.covers[self.idx], target_pose=[x, y, 0.741] + self.quat_of_target_pose, arm_tag=arm_tag, functional_point_id=0, pre_dis=0.05, dis=0.005), language_annotation=f"Put the seal back into the original lid.")

        self.info["info"] = {"keyframe_steps": list(self.keyframe_steps)}
        return self.info

    def _update_history_flags(self):
        open_height_thresh = 0.05
        for i in range(4):
            cur_pose = self.covers[i].get_pose()
            init_pose = self.cover_pose_lst[i]
            if cur_pose is not None and init_pose is not None:
                if cur_pose.p[2] > init_pose.p[2] + open_height_thresh:
                    self.cover_opened_once[i] = True

    def check_success(self):
        # update states and check blocks that should be opened according to the seal position
        self._update_history_flags()

        seal_idx = int(np.argmax(self.seal_in_cover))
        required_open_mask = [1 if i <= seal_idx else 0 for i in range(4)]

        current_open_mask = [1 if opened else 0 for opened in self.cover_opened_once]

        self.cond_open_required_covers = (current_open_mask == required_open_mask)
        
        # seal should be in the stamp area at once
        target_pos = self.target.get_pose().p
        seal_pose = self.seal.get_pose().p
        eps1 = 0.01
        con_seal_in_stamp = np.all(abs(seal_pose[:2] - target_pos[:2]) < np.array([eps1, eps1])) and seal_pose[2] < 0.84
        if con_seal_in_stamp:
            self.seal_in_stamp_once = True
        
        # seal should be back in the origin cover after stamping
        origin_cover_pose = self.covers[seal_idx].get_pose().p
        if origin_cover_pose is None:
            return False

        rel_cx = seal_pose[0] - origin_cover_pose[0]
        rel_cy = seal_pose[1] - origin_cover_pose[1]

        inside_cover_x_thresh = 0.02
        inside_cover_y_thresh = 0.02
        cond_seal_back_in_cover = (abs(rel_cx) <= inside_cover_x_thresh and abs(rel_cy) <= inside_cover_y_thresh)

        # the cover should be back to the original position after putting the seal back in
        cover_init_pose = self.cover_pose_lst[seal_idx]
        cover_back_xy_thresh = 0.08
        cover_back_z_thresh = 0.02
        cond_cover_back_to_origin = (
            abs(origin_cover_pose[0] - cover_init_pose.p[0]) <= cover_back_xy_thresh and
            abs(origin_cover_pose[1] - cover_init_pose.p[1]) <= cover_back_xy_thresh and
            abs(origin_cover_pose[2] - cover_init_pose.p[2]) <= cover_back_z_thresh
        )
        return (
            self.cond_open_required_covers and self.seal_in_stamp_once and
            cond_seal_back_in_cover and cond_cover_back_to_origin and
            self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open()
        )