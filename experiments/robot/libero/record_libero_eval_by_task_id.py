"""
List of libero goal tasks and their descriptions: * marks our tasks of interest
    Task 0: open the middle drawer of the cabinet
    *Task 1: put the bowl on the stove
    Task 2: put the wine bottle on top of the cabinet
    *Task 3: open the top drawer and put the bowl inside
    *Task 4: put the bowl on top of the cabinet
    Task 5: push the plate to the front of the stove
    *Task 6: put the cream cheese in the bowl
    *Task 7: turn on the stove
    *Task 8: put the bowl on the plate
    Task 9: put the wine bottle on the rack

Example usage (from UniVLA/experiments/robot/libero):
python record_libero_eval_by_task_id.py \
    --task_suite_name libero_goal \
    --save_video True \
    --task_id 1 \
    --target_1 akita_black_bowl_1_main \
    --target_2 flat_stove_1_burner_plate \
    --num_trials_per_task 100

python record_libero_eval_by_task_id.py \
    --task_suite_name libero_goal \
    --save_video True \
    --task_id 3 \
    --target_1 akita_black_bowl_1_main \
    --target_2 wooden_cabinet_1_cabinet_top \
    --num_trials_per_task 100

python record_libero_eval_by_task_id.py \
    --task_suite_name libero_goal \
    --save_video True \
    --task_id 4 \
    --target_1 akita_black_bowl_1_main \
    --target_2 wooden_cabinet_1_main \
    --num_trials_per_task 100

python record_libero_eval_by_task_id.py \
    --task_suite_name libero_goal \
    --save_video True \
    --task_id 6 \
    --target_1 cream_cheese_1_main \
    --target_2 akita_black_bowl_1_main \
    --num_trials_per_task 100

python record_libero_eval_by_task_id.py \
    --task_suite_name libero_goal \
    --save_video True \
    --task_id 7 \
    --target_1 flat_stove_1_button \
    --target_2 flat_stove_1_burner_plate \
    --num_trials_per_task 100

python record_libero_eval_by_task_id.py \
    --task_suite_name libero_goal \
    --save_video True \
    --task_id 8 \
    --target_1 akita_black_bowl_1_main \
    --target_2 plate_1_main \
    --num_trials_per_task 100
"""

import os
import sys
import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List

import draccus
import numpy as np
import torch
import tqdm
from record import Recording

# Append to directory so that interpreter can find detection.libero_10_object_detector.py
sys.path.append("../../../..")
from detection.libero_goal_object_relation_detector import LiberoGoalObjectRelationDetector

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
    target_1: str = ""                               # Pierrick target 1
    target_2: str = ""                               # Pierrick target 2
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                     # Number of rollouts per task
    window_size: int = 12

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


@draccus.wrap()
def eval_custom_command(cfg: GenerateConfig) -> None:
    cfg.action_decoder_path = f"../../../univla-7b-224-sft-libero/univla-libero-{cfg.task_suite_name[7:]}/action_decoder.pt"
    cfg.pretrained_checkpoint = f"../../../univla-7b-224-sft-libero/univla-libero-{cfg.task_suite_name[7:]}"

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

        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)

    # Initialize LIBERO environment and task description
    env, default_description = get_libero_env(task, cfg.model_family, resolution=256)
    # Initialize the detector if applicable. For now, it's only a libero-goal detector for task 3
    detector = None
    if cfg.task_suite_name == 'libero_goal' and cfg.task_id == 3:
        detector = LiberoGoalObjectRelationDetector(env=env.env)
    # task_description = cfg.command if cfg.command is not None else default_description
    print(f"Default task description: {default_description}")
    # cmds = cfg.commands if cfg.commands else [default_description]

    # Instantiate trajectory recorder
    recorder = Recording(env)

    # Specifically for libero-goal task 3, have one additional open-drawer recorder and a place-in-drawer recorder
    if cfg.task_suite_name == 'libero_goal' and cfg.task_id == 3:
        open_drawer_recorder = Recording(env)
        put_bowl_inside_recorder = Recording(env)

    if cfg.commands:
        # Split the string by your chosen delimiter (e.g., ';')
        # .strip() removes any leading/trailing whitespace from each command
        cmds = [cmd.strip() for cmd in cfg.commands.split(';')]
    else:
        # If --commands was not provided, use the default
        cmds = [default_description]

    resize_size = get_image_resize_size(cfg)
    latent_action_detokenize = [f"<ACT_{i}>" for i in range(32)]

    total_episodes, total_successes, total_drawer_opens = 0, 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        env.reset()
        action_decoder.reset()
        # set initial state once per episode
        obs = env.set_init_state(initial_states[episode_idx])
        # print(obs.keys())
        print(f"target 1 is {cfg.target_1}")
        print(f"target 2 is {cfg.target_2}")
        
        # Reset recorder with episodic info
        recorder.reset(
            skill_id=default_description,
            target1=cfg.target_1,
            target2=cfg.target_2
        )
        if cfg.task_suite_name == 'libero_goal' and cfg.task_id == 3:
            open_drawer_recorder.reset(
                skill_id='open the top drawer',
                target1=cfg.target_1,
                target2=cfg.target_2
            )
            put_bowl_inside_recorder.reset(
                skill_id='and put the bowl inside (the top drawer)',
                target1=cfg.target_1,
                target2=cfg.target_2
            )

        home_ee_pos = obs["robot0_eef_pos"].copy()
        home_ee_quat = obs["robot0_eef_quat"].copy()
        # convert home orientation quaternion to axis-angle
        home_ee_aa = quat2axisangle(home_ee_quat)

        episode_replay_images = []
        episode_after_drawer_open_images = []
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

        print(f"Starting episode {total_episodes+1}...")
        log_file.write(f"Starting episode {total_episodes+1}...\n")

        for task_description in cmds:
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            t = 0
            hist_action = ""
            prev_hist_action = [""]

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        if detector is not None:
                            binary_states = detector.detect_binary_states()
                        t += 1
                        num_open_drawer = 0 # for libero-goal task 3
                        continue

                    # Get preprocessed image
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
                    # Record step in trajectory
                    recorder.record_step(action)

                    if detector is not None:
                        binary_states = detector.detect_binary_states()
                        num_open_drawer += binary_states['open wooden_cabinet_1_top_region']
                        if cfg.task_suite_name == 'libero_goal' and cfg.task_id == 3:
                            if num_open_drawer <= 1:
                                open_drawer_recorder.record_step(action)
                            else:
                                episode_after_drawer_open_images.append(img)
                                put_bowl_inside_recorder.record_step(action)
                        if num_open_drawer == 1: # drawer has first become open
                            total_drawer_opens += 1
                            if cfg.save_video:
                                save_rollout_video(
                                    episode_replay_images, total_episodes, success=True, task_description='open the top drawer', log_file=log_file
                                )
                            open_drawer_recorder.save_buffer(f"./rollouts/{DATE_TIME}/", total_drawer_opens)

                    if done:
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            # Return to home position for spatial and object tasks
            if cfg.task_suite_name in ("libero_spatial", "libero_object"):
                # clear termination flag to allow further steps
                env._terminated = False
                return_to_home(env, home_ee_pos, home_ee_quat, episode_replay_images, resize_size)

            total_episodes += 1

            if cfg.save_video:
                save_rollout_video(
                    episode_replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                )
                if len(episode_after_drawer_open_images) > 0:
                    save_rollout_video(
                        episode_after_drawer_open_images, total_episodes, success=done, task_description='and put the bowl inside (the top drawer)', log_file=log_file
                    )

            # Save trajectory
            if done:
                recorder.save_buffer(f"./rollouts/{DATE_TIME}/", total_successes)
                if cfg.task_suite_name == 'libero_goal' and cfg.task_id == 3:
                    put_bowl_inside_recorder.save_buffer(f"./rollouts/{DATE_TIME}/", total_successes)
                print("Saved successful episode")

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

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
        if i%10==0:
            print(f"Home position {home_pos}")
            print(f"current pos {ee_pos}")
            print(f"step position {step_pos}")
        action = np.concatenate([step_pos, step_aa, [gripper]])
        obs, _, _, _ = env.step(action.tolist())
        # capture frame
        img = get_libero_image(obs, resize_size)
        episode_replay_images.append(img)
        if np.linalg.norm(dpos) < 1e-3 and np.linalg.norm(daa) < 1e-3:
            break


if __name__ == "__main__":
    eval_custom_command()
