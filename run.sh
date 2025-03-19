#!/bin/bash

nohup python -u run.py \
    --test_file ./data/tasks_test.jsonl \
    --api_key YOUR_GEMINI_API_KEY \
    --gemini_model gemini-2.0-flash \
    --RPM 15 \
    --max_iter 20 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --fix_box_color \
    --seed 42 > test_tasks.log &
