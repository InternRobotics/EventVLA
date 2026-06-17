from ._base_task import Base_Task
from .utils import *
import sapien
import math
import glob

class observe_and_pickup(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        shelf_pose = rand_pose(
            xlim=[0.0, 0.0],
            ylim=[0.15, 0.15],
            zlim=[0.781, 0.781],
            qpos=[0.707, 0.707, 0, 0],
        )
        self.shelf = create_actor(
            scene=self,
            pose=shelf_pose,
            modelname="008_shelf",
            convex=True,
            model_id="0",
            is_static=True,
        )
        self.shelf.set_mass(0.1)

        self.get_obs_cnt = 0

        self.object: list[Actor] = []
        self.object_modelnames = []
        self.object_ids = []
        used_pairs = set(zip(self.object_modelnames, self.object_ids))

        for i in range(5):
            def sample_valid_pose(max_outer=1000, max_inner=1000):
                for outer_step in range(max_outer):
                
                    for inner_step in range(max_inner):
                        rand_pose_cluttered = rand_pose(
                            xlim=[-0.35, 0.35],
                            ylim=[-0.25, -0.1],
                            qpos=[0.707, 0.707, 0.0, 0.0],
                            rotate_rand=True,
                            rotate_lim=[0, np.pi / 8, 0],
                        )
                        if abs(rand_pose_cluttered.p[0]) >= 0.07:
                            break
                    else:
                        return None

                    too_close = False
                    for obj in self.object:
                        peer_pose = obj.get_pose()
                        if np.linalg.norm(peer_pose.p[:2] - rand_pose_cluttered.p[:2]) < 0.08:
                            too_close = True
                            break

                    if not too_close:
                        return rand_pose_cluttered

                return None

            rand_pose_cluttered = sample_valid_pose()
            if rand_pose_cluttered is None:
                break

            def get_available_model_ids(modelname):
                asset_path = os.path.join("assets/objects", modelname)
                json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

                available_ids = []
                for file in json_files:
                    base = os.path.basename(file)
                    try:
                        idx = int(base.replace("model_data", "").replace(".json", ""))
                        available_ids.append(idx)
                    except ValueError:
                        continue

                return available_ids
            
            
            object_list = ["009_toycar", "010_mouse", "011_stapler", "012_bell", "013_playingcards"]

            max_pair_retry = 200  # 防止死循环
            picked = None
            for _ in range(max_pair_retry):
                if i == 0:
                    selected_modelname = np.random.choice(object_list)
                else:
                    if np.random.rand() < 0.2:
                        candidate_list = object_list
                    else:
                        candidate_list = [m for m in object_list if m != self.object_modelnames[0]]
                        if not candidate_list:
                            candidate_list = object_list
                    selected_modelname = np.random.choice(candidate_list)
                
                available_model_ids = get_available_model_ids(selected_modelname)
                if not available_model_ids:
                    raise ValueError(f"No available model_data.json files found for {selected_modelname}")

                selected_model_id = np.random.choice(available_model_ids)
                pair = (selected_modelname, int(selected_model_id))
                if pair in used_pairs:
                    continue

                picked = pair
                break

            if picked is None:
                break

            selected_modelname, selected_model_id = picked

            object_actor = create_actor(
                scene=self,
                pose=rand_pose_cluttered,
                modelname=selected_modelname,
                convex=True,
                model_id=selected_model_id,
            )
            self.object.append(object_actor)
            self.object_modelnames.append(selected_modelname)
            self.object_ids.append(selected_model_id)
            used_pairs.add((selected_modelname, int(selected_model_id)))

        self.target_object_idx = np.random.randint(len(self.object))
        target_object_pose = rand_pose(
            xlim=[0.0, 0.0],
            ylim=[0.15, 0.15],
            zlim=[0.88, 0.88],
            qpos=[0.707, 0.707, 0, 0],
        )
        self.target_object = create_actor(
            scene=self,
            pose=target_object_pose,
            modelname=self.object_modelnames[self.target_object_idx],
            convex=True,
            model_id=self.object_ids[self.target_object_idx], 
            is_static=True,
        )
        self.orig_left_endpose = self.get_arm_pose("left")
        self.orig_right_endpose = self.get_arm_pose("right")
        self.fail_flag = False

    def add_wall(self):
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 0.05, 0.95]),
            half_size=[0.3, 0.005, 0.2],
            color=(1, 0.9, 0.9),
            name="wall",
            is_static=True,
        )

    def play_once(self):
        self.delay(delay_time=2, save_freq=-1, language_annotation="Wait for wall appear.") # 200 / 15
        self.add_wall()
        self.delay(delay_time=1, save_freq=-1, language_annotation=f'Pick up the target object.')
        arm_tag = "left" if self.object[self.target_object_idx].get_pose().p[0] < 0 else "right"
        self.move(self.grasp_actor(self.object[self.target_object_idx], arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.02), language_annotation=f'Pick up the target object.')
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f'Pick up the target object.')
        self.get_obs_cnt = 10000
        self.info["info"] = {}
        return self.info
    
    def check_success(self):
        if self.fail_flag:
            return False
        
        self.get_obs_cnt += 1
        if self.get_obs_cnt == 20:
            self.add_wall()
        elif self.get_obs_cnt < 20:
            current_left_endpose = self.get_arm_pose("left")
            current_right_endpose = self.get_arm_pose("right")
            if np.linalg.norm(np.array(current_left_endpose[:3]) - np.array(self.orig_left_endpose[:3])) > 0.03 or \
               np.linalg.norm(np.array(current_right_endpose[:3]) - np.array(self.orig_right_endpose[:3])) > 0.03:
                self.fail_flag = True
            return False

        if self.object[self.target_object_idx].get_pose().p[2] > 0.8:
            return True
        return False