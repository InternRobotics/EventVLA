from ._base_task import Base_Task
from .utils import *

class rearrange_blocks(Base_Task):
    
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
        self.button.set_mass(0.0001, ['button_cap'])
        self.set_button_unpressed(self.button)
        self.press_cnt = 0
        self.press_flag = False

        def create_block(block_pose):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=(1,0,0),
                name="box",
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
        
        self.block_half_size = 0.02
        blocks_pose = []
        x_block = 0.02
        self.block_y_lim = np.random.uniform(-0.15, -0.08)
        for _ in range(3):
            block_pos = rand_pose(
                xlim=[x_block, x_block],
                ylim=[self.block_y_lim, self.block_y_lim],
                zlim=[0.741+self.block_half_size],
                qpos=[1, 0, 0, 0],
            )
            blocks_pose.append(block_pos)
            x_block += 0.12
        selected_mat_block = np.random.choice([0, 2])
        self.block1 = create_block(blocks_pose[1])
        self.block2 = create_block(blocks_pose[selected_mat_block])
        
        mat_half_size = [0.04, 0.04, 0.0005]
        mats_pose = []
        x_mat = 0.02
        for _ in range(2):
            mat_pos = rand_pose(
                xlim=[x_mat, x_mat],
                ylim=[self.block_y_lim, self.block_y_lim],
                zlim=[0.7415],
                qpos=[1, 0, 0, 0],
            )
            mats_pose.append(mat_pos)
            x_mat += 0.24
        self.mat1 = create_mat(mats_pose[0])
        self.mat2 = create_mat(mats_pose[1])
        if selected_mat_block == 0:
            empty_mat = 2
            self.final_occupy_mat = self.mat2
            self.mid_mat = self.mat1
        else:
            empty_mat = 0
            self.final_occupy_mat = self.mat1
            self.mid_mat = self.mat2
        self.target_pose1 = blocks_pose[empty_mat]
        self.target_pose2 = blocks_pose[1]
        self.stage_id = 0

        self.first_empty_mat_name = ['left', 'null', 'right'][empty_mat]
        self.second_block_name = ['right', 'null', 'left'][empty_mat]
    
    def play_once(self):
        self.move(self.grasp_actor(self.block1,arm_tag="right",pre_grasp_dis=0.1, grasp_dis=0.02), language_annotation=f'Pick up the {self.first_empty_mat_name} block and move it to the {self.first_empty_mat_name} empty mat.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Pick up the {self.first_empty_mat_name} block and move it to the {self.first_empty_mat_name} empty mat.')
        self.move(self.place_actor(self.block1, arm_tag="right", target_pose=self.target_pose1, functional_point_id=2, dis=0.01), language_annotation=f'Pick up the {self.first_empty_mat_name} block and move it to the {self.first_empty_mat_name} empty mat.')
        self.press_button()
        self.move(self.back_to_origin(arm_tag="left"), language_annotation=f'Press button.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Pick up the {self.second_block_name} block and move it between the two mats.')
        self.move(self.grasp_actor(self.block2,arm_tag="right",pre_grasp_dis=0.1, grasp_dis=0.02), language_annotation=f'Pick up the {self.second_block_name} block and move it between the two mats.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Pick up the {self.second_block_name} block and move it between the two mats.')
        self.move(self.place_actor(self.block2, arm_tag="right", target_pose=self.target_pose2, functional_point_id=2, dis=0.01), language_annotation=f'Pick up the {self.second_block_name} block and move it between the two mats.')
        self.info["info"] = {}
        return self.info
        
    def press_button(self):
        self.move(
            self.grasp_actor(self.button, arm_tag="left", pre_grasp_dis=0.08, grasp_dis=0.08, contact_point_id=0),
            self.back_to_origin(arm_tag="right"),
            language_annotation=f'Press button.')
        self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation=f'Press button.')
        self.check_press_success()
        self.check_success()
        self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation=f'Press button.')
        self.set_button_unpressed(self.button)
        self.update_button_reset(self.button)
    
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
        if self.stage_id == 2:
            self.max_reward = max(self.max_reward, 1.0)
            return True
        if self.press_cnt > 1:
            return False
        self.update_button_reset(self.button)
        self.check_press_success()
        self.set_button_unpressed(self.button, target=min(0.0, self.get_current_button_value("button")+0.002))
        
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        final_occupy_mat_pose = self.final_occupy_mat.get_pose().p
        mid_occupy_mat_pose = self.mid_mat.get_pose().p
        empty_region_center = [0.13, self.block_y_lim]

        if self.press_flag: # all on mat
            if self.stage_id == 0 and self.press_cnt == 1 and np.abs(block1_pose[0] - final_occupy_mat_pose[0]) < 0.03 and np.abs(block1_pose[1] - final_occupy_mat_pose[1]) < 0.03 \
                 and np.abs(block2_pose[0] - mid_occupy_mat_pose[0]) < 0.03 and np.abs(block2_pose[1] - mid_occupy_mat_pose[1]) < 0.03 and block1_pose[2] < 0.77 and block2_pose[2] < 0.77:
                self.stage_id = 1
            return False
        else:
            if self.stage_id == 1 and self.press_cnt == 1 and np.abs(block1_pose[0] - final_occupy_mat_pose[0]) < 0.03 and np.abs(block1_pose[1] - final_occupy_mat_pose[1]) < 0.03 \
                and np.abs(block2_pose[0] - empty_region_center[0]) < 0.03 and np.abs(block2_pose[1] - empty_region_center[1]) < 0.03 and self.is_right_gripper_open():
                self.stage_id = 2
                self.max_reward = max(self.max_reward, 1.0)
                return True
            return False
