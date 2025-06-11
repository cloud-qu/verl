#!/bin/bash

# 每个条目是：路径:比较符:阈值，例如：
# "ckpts/math/verl-1.5b-math-topk-noinit:<:200"
# "ckpts/math/verl-1.5b-math-ps-noinit:>:800"

CONFIGS=(
    "ckpts/math/verl-1.5b-math-topk:<:200"
    "ckpts/math/verl-1.5b-math-ps:<:200"
    "ckpts/math/verl-1.5b-math-topk-noinit:<:200"
    "ckpts/math/verl-1.5b-math-ps-noinit:<:200"
    "ckpts/math/verl-1.5b-math-dapo:<:200"
    "ckpts/math/verl-1.5b-math-dapo-thres:<:200"
    "ckpts/math/verl-1.5b-math:>:400"
    "ckpts/math/verl-7b-math:>:150"
    "ckpts/math/verl-7b-math-topk-noinit:>:180"
    "ckpts/math/verl-7b-math-topk-noinit:<:50"
    "ckpts/math/verl-7b-math-dapo:<:50"
)

for config in "${CONFIGS[@]}"; do
    IFS=":" read -r TARGET_DIR COMP_OP THRESHOLD <<< "$config"

    echo "Processing $TARGET_DIR with rule [$COMP_OP $THRESHOLD]"

    for dir in "$TARGET_DIR"/global_step_*; do
        [ -d "$dir" ] || continue

        step=$(basename "$dir" | sed 's/global_step_//')

        if [[ ! "$step" =~ ^[0-9]+$ ]]; then
            echo "Skipping $dir: invalid step"
            continue
        fi

        # 比较逻辑
        case "$COMP_OP" in
            "<")
                if [ "$step" -lt "$THRESHOLD" ]; then
                    echo "Deleting $dir (step $step < $THRESHOLD)"
                    rm -rf "$dir"
                fi
                ;;
            ">")
                if [ "$step" -gt "$THRESHOLD" ]; then
                    echo "Deleting $dir (step $step > $THRESHOLD)"
                    rm -rf "$dir"
                fi
                ;;
            *)
                echo "Unknown comparison operator: $COMP_OP"
                ;;
        esac
    done
done
