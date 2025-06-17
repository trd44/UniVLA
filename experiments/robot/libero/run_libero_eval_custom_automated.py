"""
Example usage (from project root):

python experiments/robot/libero/run_libero_eval_custom_automated.py \
    --task_suite_name libero_object \
    --save_video True \
    --use_wandb True \
    --all_tasks True \
    --num_commands 6 \
    --num_trials_per_task 1

python experiments/robot/libero/run_libero_eval_custom_automated.py \
    --task_suite_name libero_object \
    --save_video True \
    --num_commands 2 \
    --num_trials_per_task 1
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import random
import re

import draccus
import numpy as np
import torch
import tqdm

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from libero.libero import benchmark

# Reuse ActionDecoder from the default evaluation script
from experiments.robot.libero.run_libero_eval import ActionDecoder
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_latent_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


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
    # LIBERO environment-specific parameters
    ###############################################################################
    # Task suite. Options: libero_spatial, libero_object, libero_goal,
    # libero_10, libero_90
    task_suite_name: str = "libero_10"
    action_decoder_path:str = f"../../../univla-7b-224-sft-libero/univla-libero-{task_suite_name[7:]}/action_decoder.pt"
    pretrained_checkpoint: Union[str, Path] = f"../../../univla-7b-224-sft-libero/univla-libero-{task_suite_name[7:]}"     # Pretrained checkpoint path
    
    task_id: int = 0                                 # Index of the task within the suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                     # Number of rollouts per task
    window_size: int = 12

    ###############################################################################
    # Custom command
    ###############################################################################
    commands: Optional[str] = None
    num_commands: int = 1  # number of random objects to pick for commands
    basket_thresh: float = 0.08  # distance threshold to consider object in basket
    all_tasks: bool = False  # whether to run all task IDs in the suite
    all_items: bool = False  # whether to pick all items in the scene

    ###############################################################################
    # Utils
    ###############################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs
    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "UniVLA"                    # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)


@draccus.wrap()
def eval_custom_command(cfg: GenerateConfig) -> None:
    cfg.action_decoder_path = f"./univla-7b-224-sft-libero/univla-libero-{cfg.task_suite_name[7:]}/action_decoder.pt"
    cfg.pretrained_checkpoint = f"./univla-7b-224-sft-libero/univla-libero-{cfg.task_suite_name[7:]}"

    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    set_seed_everywhere(cfg.seed)

    cfg.unnorm_key = cfg.task_suite_name

    # Load action decoder
    action_decoder = ActionDecoder(cfg.window_size)
    action_decoder.net.load_state_dict(torch.load(cfg.action_decoder_path))
    action_decoder.eval().cuda()

    # Load model
    model = get_model(cfg)

    # Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    run_id = f"CUSTOM-EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        import wandb
        # Set your wandb_entity to your actual W&B username or organization, not the placeholder
        init_kwargs = {"project": cfg.wandb_project, 
                       "name": run_id, 
        }

        # only include entity if it's not the placeholder
        if cfg.wandb_entity and cfg.wandb_entity != "YOUR_WANDB_ENTITY":
            init_kwargs["entity"] = cfg.wandb_entity

        # configure system metrics sampling interval (seconds)
        settings = wandb.Settings(_stats_sampling_interval=0.5)

        wandb.init(**init_kwargs, settings=settings)


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    # determine which task IDs to run
    suite_sizes = {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_10": 10,
        "libero_90": 90,
    }
    if cfg.all_tasks:
        task_ids = list(range(suite_sizes[cfg.task_suite_name]))
    else:
        task_ids = [cfg.task_id]

    resize_size = get_image_resize_size(cfg)
    latent_action_detokenize = [f"<ACT_{i}>" for i in range(32)]

    if cfg.task_suite_name == "libero_spatial":
        max_steps = 240
    elif cfg.task_suite_name == "libero_object":
        max_steps = 300
    elif cfg.task_suite_name == "libero_goal":
        max_steps = 320
    elif cfg.task_suite_name == "libero_10":
        max_steps = 550
    elif cfg.task_suite_name == "libero_90":
        max_steps = 420
    else:
        max_steps = 300

    total_episodes, total_successes = 0, 0
    # Overall run summary counters
    overall_full_successes = 0
    overall_subtask_successes = [0] * cfg.num_commands
    for tid in task_ids:
        print(f"\n===== Starting evaluations for task_id {tid} =====")
        task = task_suite.get_task(tid)
        initial_states = task_suite.get_task_init_states(tid)
        env, default_description = get_libero_env(task, cfg.model_family, resolution=256)
        print(f"Default task description: {default_description}")
        log_file.write(f"\n\n=== Task ID: {tid} ===\n")
        log_file.write(f"Default task description: {default_description}\n")

        # Reset per-task counters for isolated metrics
        per_task_episodes, per_task_successes = 0, 0
        per_task_full_successes = 0  # count of episodes where all subtasks succeeded
        per_task_subtask_successes = [0] * cfg.num_commands

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            env.reset()
            action_decoder.reset()
            # set initial state once per episode
            obs = env.set_init_state(initial_states[episode_idx])

            # cache basket position for distance-based containment checks
            basket_pos = obs["basket_1_pos"].copy()

            home_ee_pos = obs["robot0_eef_pos"].copy()
            home_ee_quat = obs["robot0_eef_quat"].copy()
            # convert home orientation quaternion to axis-angle
            home_ee_aa = quat2axisangle(home_ee_quat)

            episode_replay_images = []

            if cfg.commands:
                # split on semicolons for explicit commands, wrap as (None, cmd)
                cmds = [(None, cmd.strip()) for cmd in cfg.commands.split(';')]
            else:
                # collect all object names except the basket from the environment
                obj_ids = env.env.obj_body_id  # dict mapping object names to body IDs
                # filter out any key that starts with 'basket'
                candidates = [name for name in obj_ids.keys() if not name.startswith("basket")]
                # determine how many to pick: either all items or cfg.num_commands
                num_to_pick = len(candidates) if cfg.all_items else cfg.num_commands
                # sample num_to_pick unique objects
                selected = random.sample(candidates, num_to_pick)
                # build English commands
                cmds = []
                for obj in selected:
                    # remove trailing number and convert underscores to spaces
                    clean_name = re.sub(r'_\d+$', '', obj).replace('_', ' ')
                    cmd_str = f"pick up the {clean_name} and put it in the basket"
                    cmds.append((obj, cmd_str))
            
            print(f"cmds {cmds}")

            print(f"Task {tid} | Episode {episode_idx+1}/{cfg.num_trials_per_task}")
            log_file.write(f"Task {tid} | Episode {episode_idx+1}/{cfg.num_trials_per_task}\n")

            print(f"Starting episode {total_episodes+1}...")

            episode_full_success = True

            # Per-command loop with new logic
            for cmd_i, (obj_name, task_description) in enumerate(cmds, start=1):
                print(f"  Subtask {cmd_i}/{len(cmds)}: {task_description}")
                log_file.write(f"  Subtask {cmd_i}/{len(cmds)}: {task_description}\n")

                cmd_success = False
                t = 0
                prev_hist_action = [""]

                # run until success (object in basket) or time runs out
                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue

                        # Get image & state, build prompt...
                        img = get_libero_image(obs, resize_size)
                        episode_replay_images.append(img)

                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        start_idx = len(prev_hist_action) if len(prev_hist_action) < 4 else 4
                        prompt_hist_action_list = [prev_hist_action[idx] for idx in range(-1 * start_idx, 0)]
                        prompt_hist_action = ""
                        for latent_action in prompt_hist_action_list:
                            prompt_hist_action += latent_action

                        latent_action, visual_embed, generated_ids = get_latent_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            hist_action=prev_hist_action[-1],
                        )

                        hist_action = ""
                        for latent_action_ids in generated_ids[0]:
                            hist_action += latent_action_detokenize[latent_action_ids.item() - 32001]
                        prev_hist_action.append(hist_action)

                        action_norm_stats = model.get_action_stats(cfg.unnorm_key)
                        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

                        action = action_decoder(latent_action, visual_embed, mask, action_low, action_high)
                        action = normalize_gripper_action(action, binarize=True)
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        obs, reward, done, info = env.step(action.tolist())

                        # check if object is in basket (for random commands)
                        if obj_name is not None:
                            obj_pos = obs[f"{obj_name}_pos"]
                            # distance check in XY plane (or full 3D)
                            if np.linalg.norm(obj_pos - basket_pos) < cfg.basket_thresh:
                                cmd_success = True
                                total_successes += 1
                                per_task_successes += 1
                                per_task_subtask_successes[cmd_i-1] += 1
                                break

                        t += 1
                    except Exception as e:
                        print(f"Caught exception: {e}")
                        log_file.write(f"Caught exception: {e}\n")
                        episode_full_success = False
                        break

                # after loop: check command outcome
                if not cmd_success:
                    episode_full_success = False
                    print(f"Task FAILED to complete in {max_steps} steps.")
                    log_file.write(f"Task FAILED: {task_description}\n")
                    break  # stop processing further commands / end episode

                # on success, optionally return home
                if cfg.task_suite_name in ("libero_spatial", "libero_object"):
                    env._terminated = False
                    return_to_home(env, home_ee_pos, home_ee_quat, episode_replay_images, resize_size)

                # record that this command succeeded
                print(f"Task SUCCEEDED: {task_description}")
                log_file.write(f"Task SUCCEEDED: {task_description}\n")

            # End of per-command loop: record episode, save video, print summary once per episode
            total_episodes += 1
            per_task_episodes += 1
            if cfg.save_video:
                # Success is True if all commands succeeded (i.e., didn't break out early)
                episode_success = (cmd_success if 'cmd_success' in locals() else False)
                save_rollout_video(
                    episode_replay_images, total_episodes, success=episode_success, task_description=task_description, log_file=log_file
                )
            if episode_full_success:
                per_task_full_successes += 1
            # log full episode result
            print(f"  Full-episode success: {episode_full_success}")
            log_file.write(f"  Full-episode success: {episode_full_success}\n")
            print(f"Success: {episode_success}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {episode_success}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Print per-task summary
        print(f"\n=== Summary for task_id {tid}: {per_task_full_successes}/{per_task_episodes} episodes fully successful, "
              f"{per_task_successes} subtask successes total "
              f"({per_task_full_successes / per_task_episodes * 100:.1f}% episodes, "
              f"{per_task_successes / (per_task_episodes*cfg.num_commands) * 100:.1f}% subtasks)")
        log_file.write(
            f"\n=== Summary for task_id {tid}: {per_task_full_successes}/{per_task_episodes} episodes fully successful, "
            f"{per_task_successes} subtask successes total "
            f"({per_task_full_successes / per_task_episodes * 100:.1f}% episodes, "
            f"{per_task_successes / (per_task_episodes*cfg.num_commands) * 100:.1f}% subtasks)\n"
        )

        # Subtask breakdown for this task
        print("Subtask successes:")
        log_file.write("Subtask successes:\n")
        for idx, count in enumerate(per_task_subtask_successes, start=1):
            print(f"  {count} command {idx}s successful")
            log_file.write(f"  {count} command {idx}s successful\n")
        # Accumulate into overall counters
        overall_full_successes += per_task_full_successes
        for i in range(cfg.num_commands):
            overall_subtask_successes[i] += per_task_subtask_successes[i]

    # Overall run summary
    print("\n=== Overall Run Summary ===")
    log_file.write("\n=== Overall Run Summary ===\n")
    print(f"Full-episode successes across all tasks: {overall_full_successes}")
    log_file.write(f"Full-episode successes across all tasks: {overall_full_successes}\n")
    print("Subtask successes across all tasks:")
    log_file.write("Subtask successes across all tasks:\n")
    for idx, count in enumerate(overall_subtask_successes, start=1):
        print(f"  {count} command {idx}s successful")
        log_file.write(f"  {count} command {idx}s successful\n")

    log_file.close()
    if cfg.use_wandb:
        import wandb

        wandb.log(
            {"success_rate/total": float(total_successes) / float(total_episodes), "num_episodes/total": total_episodes}
        )
        wandb.save(local_log_filepath)


# Helper function to return robot to home position
def return_to_home(env, home_pos, home_quat, episode_replay_images, resize_size, Kp=10.0, max_step=0.15, max_iters=150):
    """
    Moves the robot end-effector back to the given home position and orientation.
    """
    # compute desired home axis-angle
    home_aa = quat2axisangle(home_quat)
    # allow steps after a done
    env._terminated = False
    # get initial observation
    obs = env._get_obs() if hasattr(env, "_get_obs") else env.step(get_libero_dummy_action(None))[0]

    print(f"Home position {home_pos}")
    print(f"Home quaternion {home_quat}")

    def quat_inv(q):
        return np.array([-q[0], -q[1], -q[2], q[3]])
    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    for i, _ in enumerate(range(max_iters)):
        # current ee pose from latest observation
        cur_quat = obs["robot0_eef_quat"].copy()
        ee_pos   = obs["robot0_eef_pos"].copy()
        
        # compute errors
        dpos = home_pos - ee_pos
        # compute quaternion error: q_err = home_quat * inv(cur_quat)
        q_err = quat_mul(home_quat, quat_inv(cur_quat))
        err_aa = quat2axisangle(q_err)
        daa = err_aa
        # PD step (P-only)
        step_pos = np.clip(Kp * dpos, -max_step, max_step)
        step_aa = np.clip(Kp * daa, -max_step, max_step)
        step_aa = np.zeros(3)
        # always open gripper
        gripper = -1.0
        # if i%10==0:
        #     print(f"Home position {home_pos}")
        #     print(f"current pos {ee_pos}")
        #     print(f"step position {step_pos}")
        action = np.concatenate([step_pos, step_aa, [gripper]])
        obs, _, _, _ = env.step(action.tolist())
        # capture frame
        img = get_libero_image(obs, resize_size)
        episode_replay_images.append(img)
        if np.linalg.norm(dpos) < 1e-3 and np.linalg.norm(daa) < 1e-3:
            break


if __name__ == "__main__":
    eval_custom_command()
