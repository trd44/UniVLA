#!/bin/bash

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