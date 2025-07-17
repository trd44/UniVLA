"""
Example usage:
python run_libero_eval.py \
    --task_suite_name libero_10 \
    --action_decoder_path ../../../univla-7b-224-sft-libero/univla-libero-10/action_decoder.pt \
    --pretrained_checkpoint ../../../univla-7b-224-sft-libero/univla-libero-10 \
    --save_video True \
    --num_trials_per_task 1 \
    --run_id_note "my_first_libero_10_test"
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from collections import deque

import wandb

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from executor import *
import gym
from scipy.spatial.transform import Rotation as R

# Append current directory so that interpreter can find experiments.robot
#sys.path.append("../..")
# Always resolves to the project root no matter where the script is run from
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
import torch
import numpy
import numpy.core.multiarray
import pickle

# POTENTIALLY DANGEROUS: Only do this if you trust the source of the checkpoint files.
# This allows specific numpy functions/classes that are needed to load the LIBERO initial states.
# torch.serialization.add_safe_globals([
#     numpy.core.multiarray._reconstruct,
#     numpy.ndarray,
#     numpy.dtype,
#     numpy.dtypes.Float64DType,
#     pickle.UnpicklingError
# ])

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "./vla-scripts/libero_log/finetune-libero"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    
    action_decoder_path:str = "./vla-scripts/libero_log/finetune-libero/action_decoder.pt"
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    save_video: bool = True                         # Whether to save rollout videos

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"               # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                     # Number of rollouts per task
    window_size: int = 12

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs
    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

def termination_indicator(operator):
    if operator == 'pickplace':
        def Beta(state, symgoal):
            condition = state[f"on({symgoal[0]},{symgoal[1]})"] and not state[f"grasped({symgoal[0]})"]
            return condition
    else:
        def Beta(state, symgoal):
            return False
    return Beta

# Create an env wrapper which transforms the outputs of reset() and step() into gym formats (and not gymnasium formats)
class GymDiffusionWrapper(gym.Env):
    def __init__(self, env, target1, target2):
        self.env = env
        self.act_dim = 7
        high = np.inf * np.ones(self.act_dim)
        low = -high
        self.action_space = gym.spaces.Box(low, high, dtype=np.float64)
        # set up observation space
        self.obs_dim = 10

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        self.target1 = target1
        self.target2 = target2

    def reset(self):
        obs = self.env.reset()
        obs = self._get_relative_object_obs()
        return obs
    
    def set_target(self, target1, target2):
        self.target1 = target1
        self.target2 = target2

    def _get_relative_object_obs(self):
        sim = self.env.sim

        # EE pose and orientation
        #print("Get 1")
        gripper_body = sim.model.body_name2id('gripper0_eef')
        ee_pos = np.asarray(sim.data.body_xpos[gripper_body])
        ee_quat = np.asarray(sim.data.body_xquat[gripper_body])
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")

        # Object positions
        #print("Get 2")
        target1_body = sim.model.body_name2id(self.target1)
        target2_body = sim.model.body_name2id(self.target2)
        #print("Get 3")
        obj1_pos = np.asarray(sim.data.get_body_xpos(self.target1))
        obj2_pos = np.asarray(sim.data.get_body_xpos(self.target2))

        #print("Get 4")
        # Relative positions
        rel1 = obj1_pos - ee_pos
        rel2 = obj2_pos - ee_pos

        # Gripper aperture
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint1_tip")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint2_tip")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        #print(ee_euler)
        return np.concatenate([rel1, rel2, [aperture], ee_euler])

    def step(self, action):
        #action = np.concatenate([action[:4], np.asarray([0,0,0])])
        obs, reward, done, info = self.env.step(action)
        info["raw_obs"] = obs  # Store raw observation in info for debugging
        obs = self._get_relative_object_obs()

        return obs, reward, done, info

    def render(self, mode='human', *args, **kwargs):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def set_task(self, task):
        self.env.set_task(task)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    # Set random seed
    np.random.seed(cfg.seed)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{cfg.seed}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Tested Ckpt': {cfg.pretrained_checkpoint.split('/')[-1]} \n")

    # Get expected image dimensions
    resize_size = 224

    latent_action_detokenize = [f'<ACT_{i}>' for i in range(32)]

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        # Get task
        task_id = 1
        task = task_suite.get_task(task_id)

        # Load executor
        pickplace = Executor_Diffusion(id='PickPlace', 
                        policy=f"/home/hrilab/Documents/.vlas/vla-benchmarking/libero_diff_policies/18.04.13_train_diffusion_transformer_lowdim_on_stove/checkpoints/latest.ckpt",
                        I={}, 
                        Beta=termination_indicator('pickplace'),
                        nulified_action_indexes=[],
                        #oracle=True,
                        wrapper = GymDiffusionWrapper,
                        horizon=15000)
        pickplace.load_policy()

        # Define targets:
        target1 = "akita_black_bowl_1_main"
        target2 = "flat_stove_1_burner_plate"

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        n_obs_steps = 4
        n_action_steps = 8
        max_steps = 20000

        def env_fn():
            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
            # Wrap the environment
            env = GymDiffusionWrapper(env, target1, target2)
            env.set_target(target1, target2)
            env = MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
            print(f"\nTask: {task_description}")
            #log_file.write(f"\nTask: {task_description}\n")
            return env

        env_fns = [env_fn]
        dummy_env = env_fn()
        print(dummy_env.observation_space)
        obs_dim = 10
        high = np.inf * np.ones(obs_dim)
        low = -high
        observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        action_space = gym.spaces.Box(low=dummy_env.action_space.low, high=dummy_env.action_space.high, dtype=np.float64)
        print(observation_space)

        def gen_dummy_env():
            def dummy_env_fn():
                # Avoid importing or using env in the main process
                # to prevent OpenGL context issue with fork.
                # Create a fake env whose sole purpos is to provide 
                # obs/action spaces and metadata.
                env = gym.Env()
                env.observation_space = observation_space
                env.action_space = action_space
                env = GymDiffusionWrapper(env, target1, target2)
                env.metadata = {
                    'render.modes': ['human', 'rgb_array', 'depth_array'],
                    'video.frames_per_second': 12
                }
                env = MultiStepWrapper(
                    env=env,
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps,
                    max_episode_steps=max_steps
                )
                return env
            return dummy_env_fn

        print("Init env")
        env = AsyncVectorEnv(env_fns, dummy_env_fn=gen_dummy_env(), shared_memory=False)

        # Start episodes
        task_episodes, task_successes = 0, 0
        print("Starting experiment")
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):

            # Reset environment
            print("Reset env")
            env.reset()

            # Set initial states
            print("Set init state")
            #obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 240 / 8  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 300 / 8 # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 320 / 8 # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 550 / 8 # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 420 / 8 # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            success = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step([[get_libero_dummy_action(cfg.model_family),
                                                            get_libero_dummy_action(cfg.model_family),
                                                            get_libero_dummy_action(cfg.model_family),
                                                            get_libero_dummy_action(cfg.model_family)]])
                        t += 1
                        continue
                    
                    # Get preprocessed image
                    #print(info[-1]["raw_obs"][-1].keys())
                    for i in range(len(info[0]["raw_obs"])):
                        img = get_libero_image(info[0]["raw_obs"][i], resize_size)

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                    # Execute action in executor
                    obs, success, replay_images = pickplace.execute(env, obs, task, replay_images)

                    if success:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            if cfg.save_video:
                # Save a replay video of the episode
                print(len(replay_images))
                save_rollout_video(
                    replay_images, total_episodes, success=success, task_description=task_id, log_file=log_file
                )
            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_id}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_id}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
