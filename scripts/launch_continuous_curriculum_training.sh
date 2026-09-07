#!/usr/bin/env bash
set -uo pipefail

cd /home/lu/FlightLxx-IsaacLab || exit 1
export DISPLAY="${DISPLAY:-:0}"
export TERM="${TERM:-xterm-256color}"
export TOTAL_TIMESTEPS=350000
export NUM_ENVS=1024
export BUFFER_SIZE=6144
export BATCH_SIZE=24576
export NUM_UPDATES=2
export SAVE_INTERVAL=25000
export LOG_INTERVAL=100
export LEARNING_STARTS=10

./scripts/run_train.sh \
    --pretrained-checkpoint-path \
    /home/lu/FlightLxx-IsaacLab/outputs/training/Isaac-FlightLxx-CTBR-Recovery-Direct-v0/20260903_142305_seed1/checkpoints/exam_011_step_00043999.pt \
    --pretrained-actor-freeze-steps 8000 \
    --pretrained-actor-lr-ramp-steps 25000 \
    --actor-learning-rate 0.0003 \
    --actor-learning-rate-end 0.0003
exit_code=$?
printf '\nTraining exited with code %s\n' "$exit_code"
exec bash
