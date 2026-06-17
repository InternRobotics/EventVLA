from ._base_task import Base_Task
from .utils import *
from copy import deepcopy

class cover_blocks(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.cover_pose_lst = []
        self.cover_x, self.cover_y = [-0.2, 0, 0.2], [-0.05] * 3
        self.cover_name = ["left", "middle", "right"]
        for i in range(3):
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

        block_half_size = 0.02
        x_block = -0.2
        block_pose_lst = []
        for i in range(3):
            block_pose = rand_pose(
                xlim=[x_block, x_block],
                ylim=[-0.18, -0.18],
                zlim=[0.741 + block_half_size],
                qpos=[1,0,0,0],
                rotate_rand=False,
            )
            x_block += 0.2
            block_pose_lst.append(deepcopy(block_pose))

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        block_id_list = np.random.permutation([0,1,2]).tolist() 
        color_tuple = [(1,0,0), (0,1,0), (0,0,1)]
        self.blocks = []
        for i in range(len(block_pose_lst)):
            block_position, block_color = block_pose_lst[i], color_tuple[block_id_list[i]]
            self.blocks.append(create_block(block_position, block_color))

        target_state_transition = ["000", "100", "110", "111"] # first stage
        self.target_state_transition = target_state_transition
        self.open_lst = [block_id_list.index(i) for i in range(3)]
        for block_idx in self.open_lst:
            prev = self.target_state_transition[-1]
            new_state = prev[:block_idx] + "0" + prev[block_idx+1:]
            self.target_state_transition.append(new_state)
        self.reward_list = [0, 0.05, 0.1, 0.15, 0.3, 0.6, 1.0]
        self.current_state_pointer = 0
        self.fail_flag = False
        self.close_cover_palce_lst = [[block_pose.p[0], block_pose.p[1]] for block_pose in block_pose_lst]
        
    def check_if_cover(self, id):
        cover_pose = self.covers[id].get_pose().p
        block_pose = self.blocks[id].get_pose().p

        return np.abs(cover_pose[0] - block_pose[0]) < 0.03 and np.abs(cover_pose[1] - block_pose[1]) < 0.03 and (cover_pose[2] < 0.742)

    def play_once(self):
        self.last_gripper = None

        for i in range(3):
            name = ["left", "middle", "right"][i]
            arm_tag = ArmTag("left" if self.covers[i].get_pose().p[0]<0 else "right")
            if arm_tag == self.last_gripper or self.last_gripper == None:
                self.pick_and_place_cover(i)
            else:
                self.move(self.back_to_origin(arm_tag=self.last_gripper), language_annotation=self.last_annotation)
                self.pick_and_place_cover(i)
            self.update_state_transition()
        self.move(self.back_to_origin(arm_tag=self.last_gripper), language_annotation=f"Cover the {name} block with the {name} cover.")
        self.block_gripper = None
        for i in self.open_lst:
            arm_tag = ArmTag("left" if self.blocks[i].get_pose().p[0]<0 else "right")
            if arm_tag == self.block_gripper or self.block_gripper == None:
                self.open_cover(i)
            else:
                self.move(self.back_to_origin(arm_tag=self.block_gripper), language_annotation=self.last_annotation)
                self.open_cover(i)
            self.update_state_transition()
        self.move(self.back_to_origin(arm_tag=self.block_gripper), language_annotation=self.last_annotation)
        self.info["info"] = {}
        return self.info

    def pick_and_place_cover(self, cover_id, target_pose=None):
        cover = self.covers[cover_id]
        cover_pose = cover.get_pose().p
        block_pose = self.blocks[cover_id].get_pose().p
        x, y = block_pose[0], block_pose[1]
        target_pose = [x, y, 0.741]
        name = ["left", "middle", "right"][cover_id]
        self.last_annotation = f"Cover the {name} block with the {name} cover."
        arm_tag = ArmTag("left" if cover_pose[0]<0 else "right")
        self.move(self.grasp_actor(cover, arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f"Cover the {name} block with the {name} cover.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05), language_annotation=f"Cover the {name} block with the {name} cover.")
        self.move(
            self.place_actor(
                cover,
                target_pose=target_pose + self.quat_of_target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.005,
        ), language_annotation=f"Cover the {name} block with the {name} cover.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f"Cover the {name} block with the {name} cover.")
        self.last_annotation
        self.last_gripper = arm_tag
    
    def open_cover(self, block_color_id, target_pose=None):
        block_pose = self.blocks[block_color_id].get_pose().p
        x, y = block_pose[0], block_pose[1]
        target_pose = [x, y+0.13, 0.741]
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        self.move(self.grasp_actor(self.covers[block_color_id], arm_tag=arm_tag, pre_grasp_dis=0.05), language_annotation=f'Open the {self.cover_name[block_color_id]} cover to uncover the blocks in the order of red, green, and blue.')
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05), language_annotation=f'Open the {self.cover_name[block_color_id]} cover to uncover the blocks in the order of red, green, and blue.')
        self.move(
            self.place_actor(
                self.covers[block_color_id],
                target_pose=target_pose + self.quat_of_target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.005,
        ), language_annotation=f'Open the {self.cover_name[block_color_id]} cover to uncover the blocks in the order of red, green, and blue.')
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.03), language_annotation=f'Open the {self.cover_name[block_color_id]} cover to uncover the blocks in the order of red, green, and blue.')
        self.last_annotation = f'Open the {self.cover_name[block_color_id]} cover to uncover the blocks in the order of red, green, and blue.'
        self.block_gripper = arm_tag
    
    def cover_in_right_place(self):
        for i in range(3):
            cover_pose = self.covers[i].get_pose().p
            dist_xy_1 = np.linalg.norm(cover_pose[:2] - [self.cover_x[i], self.cover_y[i]])
            dist_xy_2 = np.linalg.norm(cover_pose[:2] - self.close_cover_palce_lst[i])
            if min(dist_xy_1, dist_xy_2) > 0.035 or cover_pose[2] > 0.742:
                return False
        return True

    def update_state_transition(self):
        if self.cover_in_right_place():
            state = ""
            for i in range(3):
                state += "1" if self.check_if_cover(i) else "0"
        else:
            state = "tmp"
        if state != 'tmp':
            if not self.current_state_pointer == len(self.target_state_transition) - 1:
                if state != self.target_state_transition[self.current_state_pointer]:
                    if state == self.target_state_transition[self.current_state_pointer + 1]:
                        self.current_state_pointer += 1
                    else:
                        self.fail_flag = True
            else:
                if state != self.target_state_transition[self.current_state_pointer]:
                    self.fail_flag = True
            
        self.max_reward = max(self.max_reward, self.reward_list[self.current_state_pointer])

    def check_success(self):
        if self.fail_flag:
            return False
        self.update_state_transition()
        if self.current_state_pointer == len(self.target_state_transition) - 1:
            self.max_reward = max(self.max_reward, 1.0)
            return True
        return False
