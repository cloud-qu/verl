#!/bin/bash

# 要处理的多个 ckpt 路径
TARGET_DIRS=(
    "ckpts/math/verl-1.5b-math-topk"
    "ckpts/math/verl-1.5b-math-ps"
    "ckpts/math/verl-1.5b-math-topk-noinit"
    "ckpts/math/verl-1.5b-math-ps-noinit"
    "ckpts/math/verl-1.5b-math-dapo"
    "ckpts/math/verl-1.5b-math-dapo-thres"
)

# 遍历每个目录
for TARGET_DIR in "${TARGET_DIRS[@]}"; do
    echo "Processing directory: $TARGET_DIR"
    
    for dir in "$TARGET_DIR"/global_step_*; do
        # 确保是目录
        [ -d "$dir" ] || continue

        # 提取 step 数字
        step=$(basename "$dir" | sed 's/global_step_//')

        # 如果 step 是数字并小于 200，则删除
        if [[ "$step" =~ ^[0-9]+$ ]] && [ "$step" -lt 200 ]; then
            echo "Deleting $dir (step $step)"
            # rm -rf "$dir"
        fi
    done
done
