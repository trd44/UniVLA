"""
Example usage (from project root):

python experiments/robot/libero/run_libero_eval_gr00t.py \
    --task_suite_name libero_object \
    --save_video True \
    --use_wandb True \
    --task_id 0 \
    --num_commands 6 \
    --num_trials_per_task 10

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
import tqdm
import cv2


# === GR00T integration imports ===
from gr00t.model.policy import Gr00tPolicy, EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

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

# Helper functions extracted for clarity and reuse
def init_wandb_if_needed(cfg, run_id):
    """Initialize wandb only if enabled, return the run or None."""
    if not cfg.use_wandb:
        return None
    import wandb
    init_kwargs = {"project": cfg.wandb_project, "name": run_id}
    if cfg.wandb_entity and cfg.wandb_entity != "YOUR_WANDB_ENTITY":
        init_kwargs["entity"] = cfg.wandb_entity
    settings = wandb.Settings(_stats_sampling_interval=0.5)
    return wandb.init(**init_kwargs, settings=settings)

def get_task_ids(cfg):
    """Return list of task IDs to run based on all_tasks flag."""
    suite_sizes = {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_10": 10,
        "libero_90": 90,
    }
    if cfg.all_tasks:
        return list(range(suite_sizes[cfg.task_suite_name]))
    else:
        return [cfg.task_id]

def generate_commands(env, cfg):
    """Generate list of (obj_name, command_str) tuples."""
    if cfg.commands:
        return [(None, cmd.strip()) for cmd in cfg.commands.split(';')]
    obj_ids = env.env.obj_body_id
    candidates = [name for name in obj_ids if not name.startswith("basket")]
    num_to_pick = len(candidates) if cfg.all_items else cfg.num_commands
    selected = random.sample(candidates, num_to_pick)
    cmds = []
    for obj in selected:
        clean = re.sub(r'_\d+$', '', obj).replace('_', ' ')
        cmds.append((obj, f"pick up the {clean} and place it in the basket"))
    return cmds

# Reuse ActionDecoder from the default evaluation script
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_latent_action,
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

    # (Removed: ActionDecoder and OpenVLA model loading)

    # === Instantiate GR00T policy ===
    # Attempt to select the PickNPlace data config with robust fallback
    primary_key = "robot_sim.PickNPlace"
    selected_key = primary_key
    if primary_key in DATA_CONFIG_MAP:
        gr_cfg = DATA_CONFIG_MAP[primary_key]
        selected_key = primary_key
    else:
        # Fallback: use the OXE Droid config if available
        fallback_key = "oxe_droid"
        if fallback_key in DATA_CONFIG_MAP:
            print(f"WARNING: 'robot_sim.PickNPlace' not found. Using '{fallback_key}' instead.")
            gr_cfg = DATA_CONFIG_MAP[fallback_key]
            selected_key = fallback_key
        else:
            available = ", ".join(DATA_CONFIG_MAP.keys())
            raise KeyError(
                f"No DATA_CONFIG_MAP entry for 'robot_sim.PickNPlace' or fallback '{fallback_key}'. "
                f"Available keys: {available}"
            )
    # Map selected_key to the appropriate EmbodimentTag
    if selected_key in ("single_panda_gripper", "bimanual_panda_gripper", "bimanual_panda_hand",
                        "fourier_gr1_arms_waist", "fourier_gr1_arms_only", "fourier_gr1_full_upper_body",
                        "unitree_g1", "unitree_g1_full_body"):
        emb_tag = EmbodimentTag.GR1
    elif selected_key == "oxe_droid":
        emb_tag = EmbodimentTag.OXE_DROID
    elif selected_key == "agibot_genie1":
        emb_tag = EmbodimentTag.AGIBOT_GENIE1
    else:
        # Default to NEW_EMBODIMENT for unknown configs
        print(f"WARNING: No specific EmbodimentTag for '{selected_key}', using NEW_EMBODIMENT.")
        emb_tag = EmbodimentTag.NEW_EMBODIMENT
    gr_mod_cfg = gr_cfg.modality_config()
    # Get the original transform pipeline and strip out VideoToTensor, VideoCrop, and VideoResize
    gr_mod_trans = gr_cfg.transform()
    try:
        gr_mod_trans.transforms = [
            t for t in gr_mod_trans.transforms
            if t.__class__.__name__ not in ("VideoToTensor", "VideoCrop", "VideoResize")
        ]
    except AttributeError:
        pass

    gr_policy = Gr00tPolicy(
        model_path="./GR00T-N1.5-3B",
        modality_config=gr_mod_cfg,
        modality_transform=gr_mod_trans,
        embodiment_tag=emb_tag,
        device="cuda"
    )

    run_id = f"CUSTOM-EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases if requested
    wb_run = init_wandb_if_needed(cfg, run_id)


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_ids = get_task_ids(cfg)

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

        # wrap episode index around available initial states
        num_states = len(initial_states)
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            env.reset()
            # set initial state once per episode
            state_idx = episode_idx % num_states
            obs = env.set_init_state(initial_states[state_idx])

            # cache basket position for distance-based containment checks
            basket_pos = obs["basket_1_pos"].copy()

            home_ee_pos = obs["robot0_eef_pos"].copy()
            home_ee_quat = obs["robot0_eef_quat"].copy()
            # convert home orientation quaternion to axis-angle
            home_ee_aa = quat2axisangle(home_ee_quat)

            episode_replay_images = []

            # Build commands for this episode
            cmds = generate_commands(env, cfg)
            # print(f"cmds {cmds}")
            log_file.write(f"Commands order {cmds}\n")

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
                        # Resize frame to 256×256 for GR00T video transform
                        img_resized = cv2.resize(img, (256, 256))
                        # Capture distinct camera views
                        # Front exterior view
                        front_raw = env.sim.render(camera_name="frontview", width=resize_size, height=resize_size)
                        front_rgb = cv2.cvtColor(front_raw, cv2.COLOR_BGR2RGB)
                        front_resized = cv2.resize(front_rgb, (256, 256))
                        # Side exterior view
                        side_raw = env.sim.render(camera_name="sideview", width=resize_size, height=resize_size)
                        side_rgb = cv2.cvtColor(side_raw, cv2.COLOR_BGR2RGB)
                        side_resized = cv2.resize(side_rgb, (256, 256))
                        # Ego (agent) view
                        ego_raw = env.sim.render(camera_name="agentview", width=resize_size, height=resize_size)
                        ego_rgb = cv2.cvtColor(ego_raw, cv2.COLOR_BGR2RGB)
                        ego_resized = cv2.resize(ego_rgb, (256, 256))
                        # Wrist view (already captured as wrist_resized)
                        wrist_raw = env.sim.render(camera_name="robot0_eye_in_hand", width=resize_size, height=resize_size)
                        wrist_rgb = cv2.cvtColor(wrist_raw, cv2.COLOR_BGR2RGB)
                        wrist_resized = cv2.resize(wrist_rgb, (256, 256))
                        episode_replay_images.append(img)

                        # === GR00T zero-shot inference ===
                        # Build GR00T-compatible observation (only video keys, no PIL Image)
                        gr_obs = {
                            "joint_positions": obs["robot0_eef_pos"].astype(np.float32),
                            "eef_orientation": quat2axisangle(obs["robot0_eef_quat"].astype(np.float32)),
                            "gripper": np.array([obs["robot0_gripper_qpos"]], dtype=np.float32),
                            # Distinct video views for GR00T
                            "video.exterior_image_1": np.expand_dims(front_resized, axis=0),
                            "video.exterior_image_2": np.expand_dims(side_resized, axis=0),
                            "video.ego_view":      np.expand_dims(ego_resized, axis=0),
                            "video.wrist_image":   np.expand_dims(wrist_resized, axis=0),
                        }
                        # Pass raw numpy arrays (dtype uint8, shape (T, H, W, C)) directly to gr_policy.get_action
                        # Query GR00T policy (take the first sub-action)
                        gr_action = gr_policy.get_action(gr_obs)[0]
                        action = gr_action  # use GR00T's output directly

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
