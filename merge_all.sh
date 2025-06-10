#!/bin/bash

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-topk/global_step_290/actor \
    --target_dir ckpts_send/verl-1.5b-math-topk_global_step_290_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-topk/global_step_425/actor \
    --target_dir ckpts_send/verl-1.5b-math-topk_global_step_425_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-topk/global_step_455/actor \
    --target_dir ckpts_send/verl-1.5b-math-topk_global_step_455_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-topk-noinit/global_step_275/actor \
    --target_dir ckpts_send/verl-1.5b-math-topk-noinit_global_step_275_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-topk-noinit/global_step_330/actor \
    --target_dir ckpts_send/verl-1.5b-math-topk-noinit_global_step_330_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-topk-noinit/global_step_355/actor \
    --target_dir ckpts_send/verl-1.5b-math-topk-noinit_global_step_355_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-topk-noinit/global_step_435/actor \
    --target_dir ckpts_send/verl-1.5b-math-topk-noinit_global_step_435_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-topk-noinit/global_step_545/actor \
    --target_dir ckpts_send/verl-1.5b-math-topk-noinit_global_step_545_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math/global_step_160/actor \
    --target_dir ckpts_send/verl-1.5b-math_global_step_160_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math/global_step_275/actor \
    --target_dir ckpts_send/verl-1.5b-math_global_step_275_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-ps/global_step_260/actor \
    --target_dir ckpts_send/verl-1.5b-math-ps_global_step_260_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-ps/global_step_390/actor \
    --target_dir ckpts_send/verl-1.5b-math-ps_global_step_390_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-ps-noinit/global_step_300/actor \
    --target_dir ckpts_send/verl-1.5b-math-ps-noinit_global_step_300_actor_merged && \
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ckpts/math/verl-1.5b-math-ps-noinit/global_step_520/actor \
    --target_dir ckpts_send/verl-1.5b-math-ps-noinit_global_step_520_actor_merged 