import os, zipfile, pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

class Recording:
    def __init__(self, env):
        self.env = env
        self.target1 = None
        self.target2 = None
        self.target1_id = None
        self.target1_id = None
        self.data_buffer = {}
        self.trajectory = []
        self.skill_ids = []
        self.current_skill_id = None


    def _get_relative_object_obs(self):
        sim = self.env.sim

        # EE pose and orientation
        gripper_body = sim.model.body_name2id('gripper0_eef')
        ee_pos = np.asarray(sim.data.body_xpos[gripper_body])
        ee_quat = np.asarray(sim.data.body_xquat[gripper_body])
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")

        # Object positions
        #self.target1_id = sim.model.body_name2id(self.target1)
        #self.target2_id = sim.model.body_name2id(self.target2)
        obj1_pos = np.asarray(sim.data.get_body_xpos(self.target1))
        obj1_quat = np.asarray(sim.data.get_body_xquat(self.target1))
        obj1_euler = R.from_quat(obj1_quat).as_euler("xyz")
        obj2_pos = np.asarray(sim.data.get_body_xpos(self.target2))
        obj2_quat = np.asarray(sim.data.get_body_xquat(self.target2))
        obj2_euler = R.from_quat(obj2_quat).as_euler("xyz")

        # Relative positions
        rel1 = obj1_pos - ee_pos
        rel2 = obj2_pos - ee_pos

        # Relative angles
        a_rel1 = obj1_euler - ee_euler
        a_rel2 = obj2_euler - ee_euler

        # Gripper aperture
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint1_tip")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint2_tip")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)

        return np.concatenate([rel1, a_rel1, rel2, a_rel2, [aperture]])

    def record_step(self, act):
        obs = self._get_relative_object_obs()
        self.trajectory.append(obs)
        self.trajectory.append(np.array(act))
    
    def record_detection(self, binary_states):
        self.trajectory.append(np.array(binary_states))

    def reset(self, skill_id, target1, target2):
        self.trajectory = []
        self.current_skill_id = skill_id
        self.target1 = target1
        self.target2 = target2
        if skill_id not in self.skill_ids:
            self.skill_ids.append(skill_id)
            self.data_buffer[skill_id] = []
        # Record the first observation
        first_obs = self._get_relative_object_obs()
        self.trajectory.append(first_obs)

    def get_trajectory(self):
        return self.trajectory

    def save_buffer(self, dir_path, ep_num):
        # Ensure the directory exists
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.data_buffer[self.current_skill_id].append((self.trajectory, self.target1, self.target2))

        # Decompose the data buffer into action steps
        for skill_id in self.skill_ids:
            # Convert the data buffer to bytes
            data_bytes = pickle.dumps(self.data_buffer[skill_id])
            file_path = dir_path + skill_id + f'episode_{ep_num}.zip'
            # Write the bytes to a zip file
            with zipfile.ZipFile(file_path, 'w') as zip_file:
                with zip_file.open('data.pkl', 'w', force_zip64=True) as file:
                    file.write(data_bytes)
