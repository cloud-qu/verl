set -x

data_path=/home/quy/verl/data/math/train.parquet
save_path=/home/quy/verl/data/math/train_prior.parquet
model_path=/home/quy/deepscaler/hfmodels/DeepSeek-R1-Distill-Qwen-1.5B

python3 -m recipe.ours.main_generation_prior \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=3 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=8 \
    data.batch_size=96 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.6 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=9300
