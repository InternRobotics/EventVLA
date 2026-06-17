from ._base_task import Base_Task
from .utils import *


class press_button_keyframe(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        def create_card(pose, model_id):
            return create_actor(
                scene=self,
                pose=pose,
                modelname="004_numbercard",
                is_static=True,
                model_id=model_id,
                convex=True,
            )

        # Reduce task difficulty by capping the total required presses to 5.
        self.card_id_1 = int(np.random.randint(1, 5))
        self.card_id_2 = int(np.random.randint(1, 6 - self.card_id_1))

        card_pose_1 = rand_pose(
            xlim=[-0.15, -0.15],
            ylim=[0.0, 0.0],
            qpos=[1, 0, 0, 0],
        )
        card_pose_2 = rand_pose(
            xlim=[0.02, 0.02],
            ylim=[0.0, 0.0],
            qpos=[1, 0, 0, 0],
        )
        self.card_1 = create_card(card_pose_1, self.card_id_1)
        self.card_2 = create_card(card_pose_2, self.card_id_2)

        self.button1 = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.15, -0.15],
            ylim=[-0.15, -0.15],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button1.set_mass(0.0001, ["button_cap"])
        self.set_button_unpressed(self.button1)

        self.button2 = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.0, -0.0],
            ylim=[-0.15, -0.15],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button2.set_mass(0.0001, ["button_cap"])
        self.set_button_unpressed(self.button2)

        self.check_button = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="006_check_button",
            modelid=10124,
            xlim=[0.15, 0.15],
            ylim=[-0.15, -0.15],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[0, 0, 0, 1],
            fix_root_link=True,
        )
        self.check_button.set_mass(0.0001, ["button_cap"])
        self.set_button_unpressed(self.check_button)

        self.press_cnt_1 = 0
        self.press_cnt_2 = 0
        self.press_cnt_check_button = 0
        self.press_flag_1 = False
        self.press_flag_2 = False
        self.press_flag_check_button = False
        self.keyframe_steps = []

    def _append_keyframe_step(self):
        if not self.save_data or self.FRAME_IDX <= 0:
            return

        frame_idx = int(self.FRAME_IDX - 1)
        if len(self.keyframe_steps) == 0 or self.keyframe_steps[-1] != frame_idx:
            self.keyframe_steps.append(frame_idx)

    def get_current_button_value(self, button_actor, joint_name="button_joint"):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [joint.get_name() for joint in joints]
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        return qpos[idx]

    def set_button_unpressed(self, button_actor, joint_name="button_joint", target=0.0):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [joint.get_name() for joint in joints]
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        qpos[idx] = target
        art.set_qpos(qpos)
        joints[idx].set_drive_target(target)

    def check_button_pressed(self, button_actor, joint_name="button_joint", threshold=-0.005):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [joint.get_name() for joint in joints]
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        return qpos[idx] < threshold

    def update_button_reset(self, button_actor, flag_attr, joint_name="button_joint", threshold=-0.001):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [joint.get_name() for joint in joints]
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        if qpos[idx] > threshold:
            setattr(self, flag_attr, False)

    def update_press_success(self, button_actor, flag_attr, cnt_attr):
        if self.check_button_pressed(button_actor) and not getattr(self, flag_attr):
            setattr(self, flag_attr, True)
            setattr(self, cnt_attr, getattr(self, cnt_attr) + 1)

    def press_button_once(
        self,
        button_actor,
        arm_tag,
        language_annotation,
        flag_attr,
        cnt_attr,
        press_depth,
        save_keyframe=False,
    ):
        self.move(
            self.grasp_actor(
                button_actor,
                arm_tag=arm_tag,
                pre_grasp_dis=0.08,
                grasp_dis=0.08,
                contact_point_id=0,
            ),
            language_annotation=language_annotation,
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=press_depth),
            language_annotation=language_annotation,
        )
        self.update_press_success(button_actor, flag_attr, cnt_attr)
        if save_keyframe:
            self._append_keyframe_step()
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=-press_depth),
            language_annotation=language_annotation,
        )
        self.set_button_unpressed(button_actor)
        self.update_button_reset(button_actor, flag_attr)

    def press_button_sequence(
        self,
        button_actor,
        arm_tag,
        press_times,
        language_annotation,
        flag_attr,
        cnt_attr,
        press_depth=-0.04,
        save_keyframe=False,
    ):
        for _ in range(int(press_times)):
            if not self.plan_success:
                return
            self.press_button_once(
                button_actor=button_actor,
                arm_tag=arm_tag,
                language_annotation=language_annotation,
                flag_attr=flag_attr,
                cnt_attr=cnt_attr,
                press_depth=press_depth,
                save_keyframe=save_keyframe,
            )

        if self.plan_success:
            self.move(self.back_to_origin(arm_tag=arm_tag), language_annotation=language_annotation)

    def play_once(self):
        self.keyframe_steps = []

        self.press_button_sequence(
            button_actor=self.button1,
            arm_tag="left",
            press_times=self.card_id_1,
            language_annotation="Press the left button the required number of times.",
            flag_attr="press_flag_1",
            cnt_attr="press_cnt_1",
            save_keyframe=True,
        )

        if self.plan_success:
            self.press_button_sequence(
                button_actor=self.button2,
                arm_tag="left",
                press_times=self.card_id_2,
                language_annotation="Press the middle button the required number of times.",
                flag_attr="press_flag_2",
                cnt_attr="press_cnt_2",
                save_keyframe=True,
            )

        if self.plan_success:
            self.move(self.back_to_origin(arm_tag="left"), language_annotation="Press the confirm button once.")
            self.press_button_once(
                button_actor=self.check_button,
                arm_tag="right",
                language_annotation="Press the confirm button once.",
                flag_attr="press_flag_check_button",
                cnt_attr="press_cnt_check_button",
                press_depth=-0.045,
                save_keyframe=False,
            )
            self.move(self.back_to_origin(arm_tag="right"), language_annotation="Press the confirm button once.")
            self.set_button_unpressed(self.check_button)
            self.update_button_reset(self.check_button, "press_flag_check_button")

        self.info["info"] = {
            "card_id_1": int(self.card_id_1),
            "card_id_2": int(self.card_id_2),
            "keyframe_steps": list(self.keyframe_steps),
        }
        return self.info

    def check_success(self):
        self.update_button_reset(self.button1, "press_flag_1")
        self.update_button_reset(self.button2, "press_flag_2")
        self.update_button_reset(self.check_button, "press_flag_check_button")

        self.update_press_success(self.button1, "press_flag_1", "press_cnt_1")
        self.update_press_success(self.button2, "press_flag_2", "press_cnt_2")
        self.update_press_success(self.check_button, "press_flag_check_button", "press_cnt_check_button")

        self.set_button_unpressed(
            self.button1,
            target=min(0.0, self.get_current_button_value(self.button1) + 0.002),
        )
        self.set_button_unpressed(
            self.button2,
            target=min(0.0, self.get_current_button_value(self.button2) + 0.002),
        )
        self.set_button_unpressed(
            self.check_button,
            target=min(0.0, self.get_current_button_value(self.check_button) + 0.002),
        )

        success = (
            self.press_cnt_1 == self.card_id_1
            and self.press_cnt_2 == self.card_id_2
            and self.press_cnt_check_button == 1
        )
        self.max_reward = max(self.max_reward, float(success))
        return success
