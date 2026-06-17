from ._base_task import Base_Task
from .utils import *

class swap_T(Base_Task):
    
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        rand_pos1 = rand_pose(
            xlim = [-0.09, -0.09],
            ylim = [-0.15, -0.1],
            zlim = [0.78, 0.78],
            qpos = [1,0,0,0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )
        rand_pos2 = rand_pose(
            xlim = [0.09, 0.09],
            ylim = [-0.15, -0.1],
            zlim = [0.78, 0.78],
            qpos = [1,0,0,0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        def create_T_block(T_block_pos, model_id):
            return create_actor(
                scene=self,
                pose=T_block_pos,
                modelname="007_T_block",
                model_id=model_id,
                convex=True,
            )

        self.T_block1 = create_T_block(rand_pos1, 0)
        self.T_block2 = create_T_block(rand_pos2, 1)

        def quat_mul(q1, q2):
            w1,x1,y1,z1 = q1
            w2,x2,y2,z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dtype=float)

        def normalize(q, eps=1e-12):
            n = np.linalg.norm(q)
            if n < eps:
                raise ValueError("Quaternion norm too small")
            return q / n

        def rotate_about_x_clockwise_90(q):
            q = normalize(np.asarray(q, float))
            r = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0])
            return normalize(quat_mul(q, r))
        
        self.verify_T_block1_q = self.T_block2.get_pose().q
        self.verify_T_block2_q = self.T_block1.get_pose().q

        self.target_pose1 = np.concatenate([self.T_block1.get_pose().p, rotate_about_x_clockwise_90(self.T_block1.get_pose().q)])
        self.target_pose2 = np.concatenate([self.T_block2.get_pose().p, rotate_about_x_clockwise_90(self.T_block2.get_pose().q)])
            
    def play_once(self):
        self.move(
            self.grasp_actor(self.T_block1, arm_tag="left", pre_grasp_dis=0.05, gripper_pos=0.2),
            self.grasp_actor(self.T_block2, arm_tag="right", pre_grasp_dis=0.05, gripper_pos=0.2),
            language_annotation="Pick up both T-blocks with dual arms.",
        )
        self.move(
            self.move_by_displacement(arm_tag="left", z=0.1),
            self.move_by_displacement(arm_tag="right", z=0.1),
            language_annotation="Pick up both T-blocks with dual arms.",
        )
        self.move(
            self.back_to_origin(arm_tag="left"),
            self.back_to_origin(arm_tag="right"),
            language_annotation="Move dual arms to original position.",
        )
        self.move(self.place_actor(self.T_block1, arm_tag="left", target_pose=self.target_pose2, pre_dis=0.05, constrain="align"), language_annotation="Use the left arm to place the red T-block into the initial pose of the blue T-block.")
        # self.move(self.move_by_displacement(arm_tag="left", z=0.06), language_annotation="Lift the left arm.")
        self.move(self.back_to_origin(arm_tag="left"), language_annotation="Use the left arm to place the red T-block into the initial pose of the blue T-block.")
        self.move(self.place_actor(self.T_block2, arm_tag="right", target_pose=self.target_pose1, pre_dis=0.05, constrain="align"), language_annotation="Use the right arm to place the blue T-block into the initial pose of the red T-block.")
        # self.move(self.move_by_displacement(arm_tag="right", z=0.06), language_annotation="Lift the right arm.")
        self.move(self.back_to_origin(arm_tag="right"), language_annotation="Use the right arm to place the blue T-block into the initial pose of the red T-block.")
        self.info["info"] = {}
        return self.info

    def check_success(self):
        def quat_angle_diff_rad(q, q_ref):
            """Return rotational difference angle (radians) between two quaternions.
            Quaternion order: [qx, qy, qz, qw].
            """
            q = np.asarray(q, dtype=float)
            q_ref = np.asarray(q_ref, dtype=float)

            # Normalize to be safe
            q /= (np.linalg.norm(q) + 1e-12)
            q_ref /= (np.linalg.norm(q_ref) + 1e-12)

            dot = abs(np.dot(q, q_ref))
            dot = np.clip(dot, -1.0, 1.0)
            return 2.0 * np.arccos(dot)

        # --- in your check ---
        T_block1_pose = self.T_block1.get_pose()
        T_block2_pose = self.T_block2.get_pose()

        pos1_diff = np.linalg.norm(T_block1_pose.p[:2] - self.target_pose2[:2])
        pos2_diff = np.linalg.norm(T_block2_pose.p[:2] - self.target_pose1[:2])

        angle1 = quat_angle_diff_rad(T_block1_pose.q, self.verify_T_block1_q)
        angle2 = quat_angle_diff_rad(T_block2_pose.q, self.verify_T_block2_q)

        angle_th = np.deg2rad(15.0)

        if pos1_diff < 0.025 and pos2_diff < 0.025 and \
            angle1 < angle_th and angle2 < angle_th and \
            self.is_left_gripper_open() and self.is_right_gripper_open() and T_block1_pose.p[2] < 0.77 and T_block2_pose.p[2] < 0.77:
            self.max_reward = max(self.max_reward, 1.0)
            return True
        return False