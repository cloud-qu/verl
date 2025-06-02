set -x

data_path=~/verl/data/countdown3to4/train.parquet
save_path=~/verl/data/countdown3to4/train_prior.parquet
model_path=~/verl/models/Qwen2.5-3B

python3 -m recipe.ours.main_generation_prior \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=8 \
    data.batch_size=256 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.6 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=256 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=1300 \
    reward_model.reward_manager=deepscaler
