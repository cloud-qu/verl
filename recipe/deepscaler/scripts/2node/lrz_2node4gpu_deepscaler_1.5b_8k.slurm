#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=tao.hu@lmu.de
#SBATCH -p lrz-hgx-h100-94x4

set -x

# 环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=local-66f3d1798a14c58de8f6e44c972276ff3799d7a7

# 获取节点列表并设置 Ray Head
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=26379
ip_head=$head_node:$port
export ip_head
export GLOO_SOCKET_IFNAME="ibo8.8005"

echo "Head Node IP with port: $ip_head"

source ~/.bashrc
conda activate verl2

# 启动 Ray Head 节点
echo "Starting HEAD node: $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node" --port=$port \
    --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block &

sleep 10

# 启动 Ray Worker 节点
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block &
    sleep 5
done

# 设置训练参数
export RAY_ADDRESS="http://$ip_head"
export WORKING_DIR=${PWD}
export RUNTIME_ENV="${WORKING_DIR}/verl/trainer/runtime_env.yaml"
export RAY_DATA_HOME="${WORKING_DIR}"
export MODEL_PATH="${RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
export CKPTS_DIR="${RAY_DATA_HOME}/ckpts/deepscaler/verl-deepscaler-1.5b-h100"
export TRAIN_FILE="${RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
export TEST_FILE="${RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
export NNODES=${SLURM_JOB_NUM_NODES}


project_name='deepscaler'
exp_name='verl-deepscaler-1.5b-h100-1n'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=64
n_resp_per_prompt=8

# Algorithm
temperature=0.6
val_temperature=0.6
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=64
train_micro_batch_size=64
offload=False




# 启动训练
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python3 -m recipe.deepscaler.main_deepscaler \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.gen_batch_size=${gen_prompt_bsz} \
        data.train_batch_size=${train_prompt_bsz} \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k="${top_k}" \
        actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=16 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        reward_model.reward_manager=dapo \
        reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
        reward_model.overlong_buffer.len=${overlong_buffer_len} \
        reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
        trainer.logger=['console','wandb'] \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
        trainer.nnodes=${SLURM_JOB_NUM_NODES} \
        trainer.val_before_train=False \
        trainer.test_freq=5 \
        trainer.save_freq=5 \
        trainer.total_epochs=20 \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=disable \
        tasksampler.ts_ratio=1 \
        tasksampler.framework=0 "$@"

