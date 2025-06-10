import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
    pretrained_checkpoint: Union[str, Path] = "./vla-scripts/libero_log/finetune-libero"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    action_decoder_path:str = "./vla-scripts/libero_log/finetune-libero/action_decoder.pt"
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    save_video: bool = True                          # Whether to save rollout videos

    ###############################################################################
    # LIBERO environment-specific parameters
    ###############################################################################
    # Task suite. Options: libero_spatial, libero_object, libero_goal,
    # libero_10, libero_90
    task_suite_name: str = "libero_10"
    task_id: int = 0                                 # Index of the task within the suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                     # Number of rollouts per task
    window_size: int = 12

    ###############################################################################
    # Custom command
    ###############################################################################
    command: Optional[str] = None                    # Natural language command to execute

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
    task_description = cfg.command if cfg.command is not None else default_description

    resize_size = get_image_resize_size(cfg)
    latent_action_detokenize = [f"<ACT_{i}>" for i in range(32)]

    total_episodes, total_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        print(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")

        env.reset()
        action_decoder.reset()
        hist_action = ""
        prev_hist_action = [""]
        obs = env.set_init_state(initial_states[episode_idx])

        t = 0
        replay_images = []
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
        while t < max_steps + cfg.num_steps_wait:
            try:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Get preprocessed image
                img = get_libero_image(obs, resize_size)
                replay_images.append(img)

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
                if done:
                    total_successes += 1
                    break
                t += 1
            except Exception as e:
                print(f"Caught exception: {e}")
                log_file.write(f"Caught exception: {e}\n")
                break

        total_episodes += 1

        if cfg.save_video:
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

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


if __name__ == "__main__":
    eval_custom_command()
