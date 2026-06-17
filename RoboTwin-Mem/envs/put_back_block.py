from ._base_task import Base_Task
from .utils import *

class put_back_block(Base_Task):
    
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
        self.button.set_mass(0.0001,['button_cap'])
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
        
        mat_half_size = [0.04, 0.04, 0.0005]
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
        self.mat_lst=[]
        for i in range(4):
            mat = create_mat(mats_pose[i])
            self.mat_lst.append(mat)
        self.block_half_size = 0.02
        block_id = np.random.randint(0,4)
        self.mat_name = ['left', 'right', 'front', 'back'][block_id]
        self.block = create_block(mats_pose[block_id])
        self.target_pose = self.mat_lst[block_id].get_pose().p
        self.center_pose = [0.1,-0.1,0.765,1,0,0,0]
        self.stage_id = 0
    
    def play_once(self):
        self.move(self.grasp_actor(self.block,arm_tag="right",pre_grasp_dis=0.1, grasp_dis=0.02), language_annotation=f'Pick up the block and move it to the center position.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Pick up the block and move it to the center position.')
        self.move(self.place_actor(self.block, arm_tag="right", target_pose=self.center_pose, functional_point_id=2, dis=0.02), language_annotation=f'Pick up the block and move it to the center position.')
        self.check_block_in_center()
        self.press_button()
        self.move(self.back_to_origin(arm_tag="left"), language_annotation=f'Press the button.')
        self.move(self.grasp_actor(self.block,arm_tag="right",pre_grasp_dis=0.02, grasp_dis=0.02), language_annotation=f'Move the block back to the {self.mat_name} mat.')
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Move the block back to the {self.mat_name} mat.')
        self.move(self.place_actor(self.block, arm_tag="right", target_pose=self.target_pose, functional_point_id=2, dis=0.01), language_annotation=f'Move the block back to the {self.mat_name} mat.')
        self.info["info"] = {}
        return self.info
        
    def press_button(self):
        self.move(self.grasp_actor(self.button, arm_tag="left", pre_grasp_dis=0.08, grasp_dis=0.08, contact_point_id=0), language_annotation=f'Press the button.')
        self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation=f'Press the button.')
        self.update_press_success()
        self.check_success()
        self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation=f'Press the button.')
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
    
    def update_press_success(self):
        if self.check_button_pressed(self.button) and not self.press_flag:
            self.press_flag = True
            self.press_cnt += 1

    def check_block_in_center(self):
        block_pose = self.block.get_pose().p
        return np.abs(block_pose[0] - self.center_pose[0]) < 0.04 and np.abs(block_pose[1] - self.center_pose[1]) < 0.04 and block_pose[2] < 0.77

    def check_success(self):
        if self.stage_id == 2:
            self.max_reward = max(self.max_reward, 1.0)
            return True
        self.update_button_reset(self.button)
        self.update_press_success()
        self.set_button_unpressed(self.button, target=min(0.0, self.get_current_button_value("button")+0.002))

        if self.press_flag:
            if self.stage_id == 0 and self.press_cnt == 1 and self.check_block_in_center():
                self.stage_id = 1
            return False
        else:
            if self.stage_id == 1 and self.is_right_gripper_open() and np.abs(self.block.get_pose().p[0] - self.target_pose[0]) < 0.03 \
                and np.abs(self.block.get_pose().p[1] - self.target_pose[1]) < 0.03 and self.block.get_pose().p[2] < 0.77 and self.is_right_gripper_open():
                self.stage_id = 2
                self.max_reward = max(self.max_reward, 1.0)
                return True
            return False
