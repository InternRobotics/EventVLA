from ._base_task import Base_Task
from .utils import *
import numpy as np
import colorsys

class blocks_ranking_try(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_pose_lst = []
        x_block_pose = 0.04
        for _ in range(3):
            block_pose = rand_pose(
                xlim=[x_block_pose, x_block_pose],
                ylim=[-0.1, -0.1],
                zlim=[0.765],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            block_pose_lst.append(deepcopy(block_pose))
            x_block_pose += 0.12
        
        def three_vivid_color_lst():
            h0 = float(np.random.rand())                         
            hues = (h0 + np.arange(3) / 3.0) % 1.0
            s, v = 0.9, 0.95                                    
            color_lst = [colorsys.hsv_to_rgb(float(h), s, v) for h in hues]  
            return color_lst
        
        color_lst = three_vivid_color_lst()
        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(0.02, 0.02, 0.02),
                color=color,
                name="box",
            )
        perm = np.random.permutation(3)
        while np.array_equal(perm, [0, 1, 2]):
            perm = np.random.permutation(3)

        block1_pose_id, block2_pose_id, block3_pose_id = perm

        self.block1 = create_block(block_pose_lst[block1_pose_id], color_lst[0])
        self.block2 = create_block(block_pose_lst[block2_pose_id], color_lst[1])
        self.block3 = create_block(block_pose_lst[block3_pose_id], color_lst[2])

        self.button = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.2, -0.2],
            ylim=[-0.1, -0.1],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button.set_mass(0.0001,['button_cap'])
        self.set_button_unpressed(self.button)
        self.press_cnt = 0
        self.press_flag = False

        self.perm_visited = {
            ("block1", "block2", "block3"): 0,
            ("block1", "block3", "block2"): 0,
            ("block2", "block1", "block3"): 0,
            ("block2", "block3", "block1"): 0,
            ("block3", "block1", "block2"): 0,
            ("block3", "block2", "block1"): 0,
        }
        self.repeat_perm_fail = False
        self.last_valid_perm = None

    def update_left_middle_right(self):
        blocks = [
            (self.block1, self.block1.get_pose().p[0]),
            (self.block2, self.block2.get_pose().p[0]),
            (self.block3, self.block3.get_pose().p[0]),
        ]
        blocks_sorted = sorted(blocks, key=lambda x: x[1])
        self.left_block  = blocks_sorted[0][0]
        self.middle_block = blocks_sorted[1][0]
        self.right_block  = blocks_sorted[2][0]

    def _get_block_infos(self):
        return [
            ("block1", self.block1.get_pose().p.copy()),
            ("block2", self.block2.get_pose().p.copy()),
            ("block3", self.block3.get_pose().p.copy()),
        ]

    def _get_current_valid_permutation(self, y_tol=0.04, z_tol=0.03, table_z=0.761):
        infos = self._get_block_infos()

        # get xyz
        names = [x[0] for x in infos]
        poses = [x[1] for x in infos]

        xs = np.array([p[0] for p in poses])
        ys = np.array([p[1] for p in poses])
        zs = np.array([p[2] for p in poses])

        # blocks on the table
        if not np.all(np.abs(zs - table_z) < z_tol):
            return None

        # 2) blocks roughly on the same horizontal line 
        if np.max(ys) - np.min(ys) > y_tol:
            return None

        # 3) blocks not too close to each other
        x_sorted = np.sort(xs)
        min_gap = np.min(np.diff(x_sorted))
        if min_gap < 0.04:
            return None

        # 按 x 从小到大排序，得到当前排列
        infos_sorted = sorted(infos, key=lambda x: x[1][0])
        perm = tuple([x[0] for x in infos_sorted])
        return perm
    
    def _update_permutation_status(self):
        cur_perm = self._get_current_valid_permutation()
        if cur_perm is None:
            return
        
        if cur_perm == self.last_valid_perm:
            return

        self.perm_visited[cur_perm] += 1
        self.last_valid_perm = cur_perm

        if self.perm_visited[cur_perm] >= 2:
            self.repeat_perm_fail = True

    def play_once(self):
        self.press_button()
        if self.check_success():
            self.info["info"] = {}
            return self.info
        else:
            swap_sequence = [
                ("middle", "right"),
                ("left", "right"),
                ("left", "middle"),
                ("middle", "right"),
                ("left", "right"),
            ]

            for a_name, b_name in swap_sequence:
                self.update_left_middle_right() 
                a = getattr(self, f"{a_name}_block")
                b = getattr(self, f"{b_name}_block")
                self.set_button_unpressed(self.button)
                if self.check_button_reset(self.button):
                    self.press_flag = False
                self.swap_blocks(a, b, a_name, b_name)
                self.press_button()
                if self.check_success():
                    self.info["info"] = {}
                    return self.info

    def swap_blocks(self, block_a, block_b, a_name, b_name):
        self.move(self.grasp_actor(block_a, arm_tag="right", pre_grasp_dis=0.05), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        block_a_pose_init = block_a.get_pose().p
        block_b_pose_init = block_b.get_pose().p
        temporary_pose_p = block_a.get_pose().p + np.array([0, -0.1, 0])
        temporary_pose = temporary_pose_p.tolist() + [0.707,-0.707,0,0]

        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.place_actor(block_a, arm_tag="right", target_pose=temporary_pose, pre_dis=0.1, constrain="free"), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.grasp_actor(block_b, arm_tag="right", pre_grasp_dis=0.05), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.place_actor(block_b, arm_tag="right", target_pose=block_a_pose_init.tolist() + [0.707,-0.707,0,0], pre_dis=0.1, constrain="free"), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.grasp_actor(block_a, arm_tag="right", pre_grasp_dis=0.05), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.place_actor(block_a, arm_tag="right", target_pose=block_b_pose_init.tolist() + [0.707,-0.707,0,0], pre_dis=0.1, constrain="free"), language_annotation=f'Swap the {a_name} block and the {b_name} block.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Swap the {a_name} block and the {b_name} block.')

    def press_button(self):
        self.move(self.grasp_actor(self.button, arm_tag="left", pre_grasp_dis=0.08, grasp_dis=0.08, contact_point_id=0), language_annotation=f'Press the button.')
        self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation=f'Press the button.')
        self.check_press_success()
        self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation=f'Press the button.')
    
    def get_current_button_value(self, button_name, joint_name="button_joint", target=0.0):
        if button_name == 'button':
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
        if qpos[idx] < threshold:
            return True
        else:
            return False
    
    def check_button_reset(self, button_actor, joint_name="button_joint", threshold=-0.001): 
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor   
        joints = art.get_active_joints()   
        joint_names = [j.get_name() for j in joints]    
        idx = joint_names.index(joint_name) 
        qpos = art.get_qpos()
        if qpos[idx] > threshold:
            self.press_flag = False
            return True
        else:
            return False
    
    def check_press_success(self):
        if self.check_button_pressed(self.button) and not self.press_flag:
            self.press_flag = True
            self.press_cnt += 1

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p

        eps = [0.13, 0.04]

        self.check_press_success()
        self.set_button_unpressed(self.button, target=min(0.0, self.get_current_button_value("button")+0.002))
        self.check_button_reset(self.button)

        self._update_permutation_status()
        # print(self.repeat_perm_fail)
        if self.repeat_perm_fail:
            return False

        return (np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
                and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps) and block1_pose[0] < block2_pose[0]
                and block2_pose[0] < block3_pose[0] and self.is_right_gripper_open()
                and self.press_flag)
    