#!/bin/bash

nohup python -u run.py \
    --test_file ./data/tasks_test.jsonl \
    --api_key  \
    --gemini_model gemini-2.0-flash \
    --RPM 15 \
    --activate_EGA \
    --trajectory \
    --error_max_reflection_iter 3 \
    --max_iter 20 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --fix_box_color \
    --start_maximized \
    --seed 42 > test_tasks.log &
