# train_wrapper.py

import argparse
import os

from hydra import compose, initialize
from omegaconf import OmegaConf

from recipe.deepscaler.main_deepscaler import main


def main_wrapper_deepscaler_uniform():
    # 设置训练参数
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/deepscaler/verl-deepscaler-1.5b-h100"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-2n'
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.28

    max_prompt_length=1024
    max_response_length=1024 * 8
    enable_overlong_buffer=False
    overlong_buffer_len=1024 * 4
    overlong_penalty_factor=1.0

    loss_agg_mode="token-mean"

    enable_filter_groups=False
    filter_groups_metric='acc'
    max_num_gen_batches=10
    train_prompt_bsz=128
    gen_prompt_bsz=train_prompt_bsz
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

    with initialize(config_path="verl/trainer/config", version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.gen_batch_size={gen_prompt_bsz}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    f"algorithm.adv_estimator={adv_estimator}",
                    f"algorithm.use_kl_in_reward={use_kl_in_reward}",
                    f"algorithm.kl_ctrl.kl_coef={kl_coef}",
                    f"algorithm.filter_groups.enable={enable_filter_groups}",
                    f"algorithm.filter_groups.metric={filter_groups_metric}",
                    f"algorithm.filter_groups.max_num_gen_batches={max_num_gen_batches}",
                    "actor_rollout_ref.model.use_remove_padding=True",
                    f"actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0",
                    "actor_rollout_ref.actor.grad_clip=1.0",
                    f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
                    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
                    "actor_rollout_ref.rollout.gpu_memory_utilization=0.85",
                    f"actor_rollout_ref.rollout.log_prob_micro_batch_size={infer_micro_batch_size}",
                    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                    "actor_rollout_ref.rollout.enable_chunked_prefill=True",
                    f"actor_rollout_ref.rollout.max_num_batched_tokens={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.rollout.temperature={temperature}",
                    f"actor_rollout_ref.rollout.top_p={top_p}",
                    f"actor_rollout_ref.rollout.top_k={top_k}",
                    f"actor_rollout_ref.rollout.val_kwargs.temperature={val_temperature}",
                    f"actor_rollout_ref.rollout.val_kwargs.top_p={top_p}",
                    f"actor_rollout_ref.rollout.val_kwargs.top_k={top_k}",
                    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
                    "actor_rollout_ref.rollout.val_kwargs.n=16",
                    f"actor_rollout_ref.ref.log_prob_micro_batch_size={infer_micro_batch_size}",
                    f"actor_rollout_ref.ref.fsdp_config.param_offload={offload}",
                    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=1",
                    "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1",
                    "reward_model.reward_manager=deepscaler",
                    f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                    f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                    f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                    "trainer.logger=['console','wandb']",
                    f"trainer.project_name={project_name}",
                    f"trainer.experiment_name={exp_name}",
                    "trainer.n_gpus_per_node=4",
                    "trainer.nnodes=2",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "tasksampler.ts_ratio=1",
                    "tasksampler.framework=0",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_deepscaler_topk():
    # 设置训练参数
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/deepscaler/verl-deepscaler-1.5b-h100"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/deepscaler/data/index_score.json"
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-topk-2n'
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.28

    max_prompt_length=1024
    max_response_length=1024 * 8
    enable_overlong_buffer=False
    overlong_buffer_len=1024 * 4
    overlong_penalty_factor=1.0

    loss_agg_mode="token-mean"

    enable_filter_groups=False
    filter_groups_metric='acc'
    max_num_gen_batches=10
    train_prompt_bsz=128
    gen_prompt_bsz=train_prompt_bsz
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

    with initialize(config_path="verl/trainer/config", version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.gen_batch_size={gen_prompt_bsz}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    f"algorithm.adv_estimator={adv_estimator}",
                    f"algorithm.use_kl_in_reward={use_kl_in_reward}",
                    f"algorithm.kl_ctrl.kl_coef={kl_coef}",
                    f"algorithm.filter_groups.enable={enable_filter_groups}",
                    f"algorithm.filter_groups.metric={filter_groups_metric}",
                    f"algorithm.filter_groups.max_num_gen_batches={max_num_gen_batches}",
                    "actor_rollout_ref.model.use_remove_padding=True",
                    f"actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0",
                    "actor_rollout_ref.actor.grad_clip=1.0",
                    f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
                    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
                    "actor_rollout_ref.rollout.gpu_memory_utilization=0.85",
                    f"actor_rollout_ref.rollout.log_prob_micro_batch_size={infer_micro_batch_size}",
                    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                    "actor_rollout_ref.rollout.enable_chunked_prefill=True",
                    f"actor_rollout_ref.rollout.max_num_batched_tokens={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.rollout.temperature={temperature}",
                    f"actor_rollout_ref.rollout.top_p={top_p}",
                    f"actor_rollout_ref.rollout.top_k={top_k}",
                    f"actor_rollout_ref.rollout.val_kwargs.temperature={val_temperature}",
                    f"actor_rollout_ref.rollout.val_kwargs.top_p={top_p}",
                    f"actor_rollout_ref.rollout.val_kwargs.top_k={top_k}",
                    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
                    "actor_rollout_ref.rollout.val_kwargs.n=16",
                    f"actor_rollout_ref.ref.log_prob_micro_batch_size={infer_micro_batch_size}",
                    f"actor_rollout_ref.ref.fsdp_config.param_offload={offload}",
                    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=1",
                    "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1",
                    "reward_model.reward_manager=deepscaler",
                    f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                    f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                    f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                    "trainer.logger=['console','wandb']",
                    f"trainer.project_name={project_name}",
                    f"trainer.experiment_name={exp_name}",
                    "trainer.n_gpus_per_node=4",
                    "trainer.nnodes=2",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "ttasksampler.ts_ratio=16",
                    "tasksampler.framework=4",
                    "tasksampler.bandit_sample_strategy='topk'",
                    "tasksampler.bandit_init=True",
                    f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)

def main_wrapper_deepscaler_ps():
    # 设置训练参数
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/deepscaler/verl-deepscaler-1.5b-h100"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/deepscaler/data/index_score.json"
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-ps-2n'
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.28

    max_prompt_length=1024
    max_response_length=1024 * 8
    enable_overlong_buffer=False
    overlong_buffer_len=1024 * 4
    overlong_penalty_factor=1.0

    loss_agg_mode="token-mean"

    enable_filter_groups=False
    filter_groups_metric='acc'
    max_num_gen_batches=10
    train_prompt_bsz=128
    gen_prompt_bsz=train_prompt_bsz
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

    with initialize(config_path="verl/trainer/config", version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.gen_batch_size={gen_prompt_bsz}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    f"algorithm.adv_estimator={adv_estimator}",
                    f"algorithm.use_kl_in_reward={use_kl_in_reward}",
                    f"algorithm.kl_ctrl.kl_coef={kl_coef}",
                    f"algorithm.filter_groups.enable={enable_filter_groups}",
                    f"algorithm.filter_groups.metric={filter_groups_metric}",
                    f"algorithm.filter_groups.max_num_gen_batches={max_num_gen_batches}",
                    "actor_rollout_ref.model.use_remove_padding=True",
                    f"actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0",
                    "actor_rollout_ref.actor.grad_clip=1.0",
                    f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
                    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
                    "actor_rollout_ref.rollout.gpu_memory_utilization=0.85",
                    f"actor_rollout_ref.rollout.log_prob_micro_batch_size={infer_micro_batch_size}",
                    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                    "actor_rollout_ref.rollout.enable_chunked_prefill=True",
                    f"actor_rollout_ref.rollout.max_num_batched_tokens={max_prompt_length + max_response_length}",
                    f"actor_rollout_ref.rollout.temperature={temperature}",
                    f"actor_rollout_ref.rollout.top_p={top_p}",
                    f"actor_rollout_ref.rollout.top_k={top_k}",
                    f"actor_rollout_ref.rollout.val_kwargs.temperature={val_temperature}",
                    f"actor_rollout_ref.rollout.val_kwargs.top_p={top_p}",
                    f"actor_rollout_ref.rollout.val_kwargs.top_k={top_k}",
                    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
                    "actor_rollout_ref.rollout.val_kwargs.n=16",
                    f"actor_rollout_ref.ref.log_prob_micro_batch_size={infer_micro_batch_size}",
                    f"actor_rollout_ref.ref.fsdp_config.param_offload={offload}",
                    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=1",
                    "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1",
                    "reward_model.reward_manager=deepscaler",
                    f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                    f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                    f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                    "trainer.logger=['console','wandb']",
                    f"trainer.project_name={project_name}",
                    f"trainer.experiment_name={exp_name}",
                    "trainer.n_gpus_per_node=4",
                    "trainer.nnodes=2",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "ttasksampler.ts_ratio=16",
                    "tasksampler.framework=4",
                    "tasksampler.bandit_sample_strategy='threshold'",
                    "tasksampler.bandit_lower_bound=0.3",
                    "tasksampler.bandit_upper_bound=0.7",
                    "tasksampler.bandit_init=True",
                    f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)

if __name__ == "__main__":
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    print(f"slurm_job_id: {slurm_job_id}")
    slurm_job_id = str(slurm_job_id)

    main_wrapper_deepscaler_uniform()
