# eval_hanoi_finetune.py

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import tqdm
import draccus
import imageio

sys.path.append("../..")
sys.path.insert(0, "/home/hrilab/Documents/.vlas/vla-benchmarking/robosuite/src")
# sys.path.insert(0, "/home/hrilab/Documents/.vlas/cycliclxm-slim/CyclicLxM")

print("sys.executable:", sys.executable)
print("sys.path[0]:", sys.path[0])
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.hanoi import Hanoi
# from dataset_making.record_demos import RecordDemos

from experiments.robot.libero.run_libero_eval import ActionDecoder
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_image_resize_size,
    get_latent_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)

# ================== CONFIG (copied from run_libero_eval_custom.py) ====================

@dataclass
class GenerateConfig:
    # fmt: off

    ###############################################################################
    # Model-specific parameters
    ###############################################################################
    model_family: str = "openvla"                    # Model family
    # pretrained_checkpoint: Union[str, Path] = "./vla-scripts/libero_log/finetune-libero"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    # action_decoder_path:str = "./vla-scripts/libero_log/finetune-libero/action_decoder.pt"
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    save_video: bool = True                          # Whether to save rollout videos

    ###############################################################################
    # LIBERO environment-specific parameters (ignored in Hanoi)
    ###############################################################################
    task_suite_name: str = "libero_10"
    action_decoder_path:str = "/home/hrilab/Documents/.vlas/vla-benchmarking/UniVLA/vla-scripts/runs/univla-7b+hanoi_full+b1+lr-0.00035+lora-r32+dropout-0.0--image_aug=w-LowLevelDecoder-ws-1/action_decoder-30000.pt"
    pretrained_checkpoint: Union[str, Path] = "/home/hrilab/Documents/.vlas/vla-benchmarking/UniVLA/vla-scripts/runs/univla-7b+hanoi_full+b1+lr-0.00035+lora-r32+dropout-0.0--image_aug=w-LowLevelDecoder-ws-1"
    unnorm_key:str = "hanoi_full"
    
    task_id: int = 0                                 # Index of the task within the suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                     # Number of rollouts per task
    window_size: int = 1

    ###############################################################################
    # Custom command
    ###############################################################################
    commands: Optional[str] = None

    ###############################################################################
    # Utils
    ###############################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs
    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    ###############################################################################
    # HANOI-specific add-ons
    ###############################################################################
    max_steps: int = 500
    save_gif: bool = True
    gif_path: str = "hanoi_eval.gif"
    task_description: Optional[str] = "Pick up the blue block"

    # fmt: on

# ================== END CONFIG ====================

def make_hanoi_env(render=False):
    ctrl_cfg = suite.load_controller_config(default_controller='OSC_POSE')
    env = suite.make(
        "Hanoi",
        robots="Panda",
        controller_configs=ctrl_cfg,
        has_renderer=render,
        has_offscreen_renderer=True,
        horizon=1000,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        random_reset=False
    )
    return env

def get_hanoi_image(obs, size=(256, 256)):
    img = obs.get("agentview_image")
    if img is None:
        raise RuntimeError("No agentview_image in obs")
    img = np.asarray(img, dtype=np.uint8)
    if size is not None and img.shape[:2] != size:
        # import cv2
        # img = cv2.resize(img, size[::-1])
        img = get_libero_image(obs, 224)
    return img

@draccus.wrap()
def eval_hanoi_finetune(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)

    print(f"Loading action decoder from: {cfg.action_decoder_path}")
    action_decoder = ActionDecoder(cfg.window_size)
    action_decoder.net.load_state_dict(torch.load(cfg.action_decoder_path, map_location="cpu"))
    action_decoder.eval().cuda()

    print(f"Loading model from: {cfg.pretrained_checkpoint}")
    model = get_model(cfg)
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    resize_size = get_image_resize_size(cfg)
    latent_action_detokenize = [f"<ACT_{i}>" for i in range(32)]

    env = make_hanoi_env(render=False)

    # For multiple episodes (for consistency with LIBERO script)
    total_episodes, total_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        obs = env.reset()
        frames = []
        done = False
        t = 0
        prev_hist_action = [""]

        # Use either provided command or default
        if cfg.commands:
            cmds = [cmd.strip() for cmd in cfg.commands.split(';')]
        else:
            cmds = [cfg.task_description]

        for task_description in cmds:
            print(f"\nTask: {task_description}")

            while not done and t < cfg.max_steps:
                # Save frame for GIF
                frame = env.sim.render(width=640, height=480, camera_name="agentview")
                frames.append(frame)

                img = get_hanoi_image(obs, resize_size)
                observation = {
                    "full_image": img,
                    "state": np.concatenate([
                        obs["robot0_eef_pos"],
                        obs["robot0_eef_quat"],
                        obs["robot0_gripper_qpos"],
                    ]),
                }

                start_idx = len(prev_hist_action) if len(prev_hist_action) < 4 else 4
                prompt_hist_action_list = [prev_hist_action[idx] for idx in range(-1 * start_idx, 0)]
                prompt_hist_action = "".join(prompt_hist_action_list)

                with torch.no_grad():
                    latent_action, visual_embed, generated_ids = get_latent_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        hist_action=prev_hist_action[-1],
                    )

                hist_action = "".join([
                    latent_action_detokenize[latent_action_ids.item() - 32001]
                    for latent_action_ids in generated_ids[0]
                ])
                prev_hist_action.append(hist_action)

                action_norm_stats = model.get_action_stats(cfg.unnorm_key)
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
                action = action_decoder(latent_action, visual_embed, mask, action_low, action_high)
                action = normalize_gripper_action(action, binarize=True)
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)

                obs, reward, done, info = env.step(action.tolist())
                t += 1

            total_episodes += 1
            if done:
                total_successes += 1
            print(f"Episode finished in {t} steps. Success: {done}")

        # Save as GIF
        if cfg.save_gif:
            save_rollout_video(
                frames, total_episodes, success=done, task_description=task_description
            )
            # gif_path = cfg.gif_path if cfg.num_trials_per_task == 1 else f"hanoi_eval_{episode_idx:03d}.gif"
            # imageio.mimsave(gif_path, frames, fps=10)
            # print(f"Saved GIF to {gif_path}")

    print(f"Successes: {total_successes}/{total_episodes}")

if __name__ == "__main__":
    eval_hanoi_finetune()