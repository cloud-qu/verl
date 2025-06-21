# train_wrapper.py

import argparse
import os

from hydra import compose, initialize
from omegaconf import OmegaConf



def main_wrapper_deepscaler_uniform(is_debug=False):
    from recipe.deepscaler.main_deepscaler_remote import main
    if is_debug:
        nnodes = 2
        n_gpus_per_node=2
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-2n'
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.2

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
    infer_micro_batch_size=128
    train_micro_batch_size=64
    offload=False

    with initialize(config_path="recipe/deepscaler/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                    f"trainer.n_gpus_per_node={n_gpus_per_node}",
                    f"trainer.nnodes={nnodes}",
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


def main_wrapper_deepscaler_topk(is_debug=False):
    from recipe.deepscaler.main_deepscaler_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=4
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-topk-2n'
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/deepscaler/data/index_score.json"
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.2

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
    infer_micro_batch_size=128
    train_micro_batch_size=64
    offload=False

    with initialize(config_path="recipe/deepscaler/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                    f"trainer.n_gpus_per_node={n_gpus_per_node}",
                    f"trainer.nnodes={nnodes}",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "tasksampler.ts_ratio=16",
                    "tasksampler.framework=4",
                    "tasksampler.bandit_sample_strategy='topk'",
                    "tasksampler.bandit_init=True",
                    f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)

def main_wrapper_deepscaler_ps(is_debug=False):
    from recipe.deepscaler.main_deepscaler_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-ps-2n'
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/deepscaler/data/index_score.json"
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.2

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
    infer_micro_batch_size=128
    train_micro_batch_size=64
    offload=False

    with initialize(config_path="recipe/deepscaler/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=28000",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                    f"trainer.n_gpus_per_node={n_gpus_per_node}",
                    f"trainer.nnodes={nnodes}",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "tasksampler.ts_ratio=16",
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

def main_wrapper_deepscaler_ps_noinit(is_debug=False):
    from recipe.deepscaler.main_deepscaler_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-ps-noinit-2n'
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/deepscaler/data/index_score.json"
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.2

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
    infer_micro_batch_size=128
    train_micro_batch_size=64
    offload=False

    with initialize(config_path="recipe/deepscaler/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                    f"trainer.n_gpus_per_node={n_gpus_per_node}",
                    f"trainer.nnodes={nnodes}",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "tasksampler.ts_ratio=16",
                    "tasksampler.framework=4",
                    "tasksampler.bandit_sample_strategy='threshold'",
                    "tasksampler.bandit_lower_bound=0.3",
                    "tasksampler.bandit_upper_bound=0.7",
                    "tasksampler.bandit_init=False",
                    f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_deepscaler_topk_noinit(is_debug=False):
    from recipe.deepscaler.main_deepscaler_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-topk-noinit-2n'
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/deepscaler/data/index_score.json"
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.2

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
    infer_micro_batch_size=128
    train_micro_batch_size=64
    offload=False

    with initialize(config_path="recipe/deepscaler/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                    f"trainer.n_gpus_per_node={n_gpus_per_node}",
                    f"trainer.nnodes={nnodes}",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "tasksampler.ts_ratio=16",
                    "tasksampler.framework=4",
                    "tasksampler.bandit_sample_strategy='topk'",
                    "tasksampler.bandit_init=False",
                    f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)



def main_wrapper_deepscaler_srpo(is_debug=False):
    from recipe.deepscaler.main_deepscaler_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-srpo-2n'
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/deepscaler/data/index_score.json"
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.2

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
    infer_micro_batch_size=128
    train_micro_batch_size=64
    offload=False

    with initialize(config_path="recipe/deepscaler/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                    f"trainer.n_gpus_per_node={n_gpus_per_node}",
                    f"trainer.nnodes={nnodes}",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "tasksampler.ts_ratio=1",
                    "tasksampler.framework=5",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_deepscaler_ps_upperdecay(is_debug=False):
    from recipe.deepscaler.main_deepscaler_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    WORKING_DIR=os.path.dirname(os.path.abspath(__file__))
    project_name='deepscaler'
    exp_name='verl-deepscaler-1.5b-h100-ps-upperdecay-2n'
    RAY_DATA_HOME=f"{WORKING_DIR}"
    MODEL_PATH=f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR=f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/train.parquet"
    TEST_FILE=f"{RAY_DATA_HOME}/recipe/deepscaler/processed_data/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/deepscaler/data/index_score.json"
    adv_estimator='grpo'

    use_kl_in_reward=False
    kl_coef=0.001
    use_kl_loss=True
    kl_loss_coef=0.001

    clip_ratio_low=0.2
    clip_ratio_high=0.2

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
    infer_micro_batch_size=128
    train_micro_batch_size=64
    offload=False

    with initialize(config_path="recipe/deepscaler/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                    f"data.train_files='{TRAIN_FILE}'",
                    f"data.val_files='{TEST_FILE}'",
                    "data.prompt_key=prompt",
                    "data.truncation='left'",
                    f"data.max_prompt_length={max_prompt_length}",
                    f"data.max_response_length={max_response_length}",
                    f"data.train_batch_size={train_prompt_bsz}",
                    f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                    f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                    f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                    f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                    f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                    "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=25000",
                    f"actor_rollout_ref.model.path={MODEL_PATH}",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
                    "actor_rollout_ref.actor.optim.weight_decay=0.1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                    f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                    "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                    f"trainer.n_gpus_per_node={n_gpus_per_node}",
                    f"trainer.nnodes={nnodes}",
                    "trainer.val_before_train=False",
                    "trainer.test_freq=5",
                    "trainer.save_freq=5",
                    "trainer.total_epochs=20",
                    f"trainer.default_local_dir={CKPTS_DIR}",
                    "trainer.resume_mode=disable",
                    "tasksampler.ts_ratio=1",
                    "tasksampler.framework=4",
                    "tasksampler.bandit_sample_strategy='threshold'",
                    "tasksampler.bandit_lower_bound=0.3",
                    "tasksampler.bandit_upper_bound=0.7",
                    "tasksampler.bandit_init=True",
                    f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
                    "tasksampler.bandit_upper_decay_steps=700",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)
    

def main_wrapper_math_uniform(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-1.5b-math'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=1",
                "tasksampler.framework=0",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_math_topk(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-1.5b-math-topk'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/ours/scripts/math/index_score.json"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=16",
                "tasksampler.framework=4",
                "tasksampler.bandit_sample_strategy='topk'",
                "tasksampler.bandit_init=True",
                f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_math_ps(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-1.5b-math-ps'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/ours/scripts/math/index_score.json"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=16",
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


def main_wrapper_math_topk_noinit(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-1.5b-math-topk-noinit'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/ours/scripts/math/index_score.json"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=16",
                "tasksampler.framework=4",
                "tasksampler.bandit_sample_strategy='topk'",
                "tasksampler.bandit_init=False",
                f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_math_ps_noinit(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-1.5b-math-ps-noinit'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/ours/scripts/math/index_score.json"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=16",
                "tasksampler.framework=4",
                "tasksampler.bandit_sample_strategy='threshold'",
                "tasksampler.bandit_lower_bound=0.3",
                "tasksampler.bandit_upper_bound=0.7",
                "tasksampler.bandit_init=False",
                f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_math_dapo(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-1.5b-math-dapo'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/ours/scripts/math/index_score.json"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = True
    filter_groups_metric = 'score'
    max_num_gen_batches = 4

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=100",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=auto",
                "tasksampler.ts_ratio=1",
                "tasksampler.framework=0",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_math_dapo_thres(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-1.5b-math-dapo-thres'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/ours/scripts/math/index_score.json"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = True
    filter_groups_metric = 'score'
    max_num_gen_batches = 4

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
                f"algorithm.adv_estimator={adv_estimator}",
                f"algorithm.use_kl_in_reward={use_kl_in_reward}",
                f"algorithm.kl_ctrl.kl_coef={kl_coef}",
                f"algorithm.filter_groups.enable={enable_filter_groups}",
                f"algorithm.filter_groups.metric={filter_groups_metric}",
                f"algorithm.filter_groups.max_num_gen_batches={max_num_gen_batches}",
                f"algorithm.filter_groups.filter_min=0.3",
                f"algorithm.filter_groups.filter_max=0.7",
                "actor_rollout_ref.model.use_remove_padding=True",
                f"actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz}",
                f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
                f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=100",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=auto",
                "tasksampler.ts_ratio=1",
                "tasksampler.framework=0",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_math_7b_uniform(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-7b-math'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-7B"
    if not os.path.exists(MODEL_PATH):
        os.system("huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir models/DeepSeek-R1-Distill-Qwen-7B")
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 2}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=auto",
                "tasksampler.ts_ratio=1",
                "tasksampler.framework=0",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)



def main_wrapper_math_7b_topk_noinit(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-7b-math-topk-noinit'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-7B"
    if not os.path.exists(MODEL_PATH):
        os.system("huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir models/DeepSeek-R1-Distill-Qwen-7B")
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 2}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path='{RAY_DATA_HOME}/ckpts/math/verl-7b-math-topk-noinit/global_step_105'",
                "tasksampler.ts_ratio=16",
                "tasksampler.framework=4",
                "tasksampler.bandit_sample_strategy='topk'",
                "tasksampler.bandit_init=False",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_math_7b_dapo(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=4
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-7b-math-dapo'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"WORKING_DIR: {WORKING_DIR}")
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-7B"
    if not os.path.exists(MODEL_PATH):
        os.system("huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir models/DeepSeek-R1-Distill-Qwen-7B")
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = True
    filter_groups_metric = 'score'
    max_num_gen_batches = 4

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 2}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=100",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=auto",
                "tasksampler.ts_ratio=1",
                "tasksampler.framework=0",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)



def main_wrapper_math_topk_offline(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-1.5b-math-topk-offline'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    BANDIT_INIT_PATH=f"{RAY_DATA_HOME}/recipe/ours/scripts/math/index_score.json"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=16",
                "tasksampler.framework=4",
                "tasksampler.bandit_sample_strategy='topk'",
                "tasksampler.bandit_no_update=True",
                "tasksampler.bandit_init=True",
                f"tasksampler.bandit_init_dir={BANDIT_INIT_PATH}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)




def main_wrapper_math_7b_topk_noinit_new(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'math'
    exp_name = 'verl-7b-math-topk-noinit-new'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-7B"
    if not os.path.exists(MODEL_PATH):
        os.system("huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir models/DeepSeek-R1-Distill-Qwen-7B")
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/math/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/deepscaler/aime.parquet"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/math_dataset.py --local_dir='data/math'")
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024 * 8
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = True
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 2}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=5",
                "trainer.save_freq=5",
                "trainer.total_epochs=40",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=16",
                "tasksampler.framework=4",
                "tasksampler.bandit_sample_strategy='topk'",
                "tasksampler.bandit_init=False",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)

def main_wrapper_geo_7b_uniform(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 2
        n_gpus_per_node=2
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'geo3k'
    exp_name = 'verl-7b-geo'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/Qwen2.5-VL-7B-Instruct"
    # MODEL_PATH = f"Qwen/Qwen2.5-VL-7B-Instruct"
    if not os.path.exists(MODEL_PATH):
        os.system("huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct")
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/geo3k/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/geo3k/test.parquet"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/geo3k.py --local_dir='data/geo3k'")
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = False
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 2}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 4}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
                "actor_rollout_ref.actor.grad_clip=1.0",
                f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
                "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=10",
                "trainer.save_freq=10",
                "trainer.max_actor_ckpt_to_keep=5",
                "trainer.total_epochs=400",
                "trainer.total_training_steps=120",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=1",
                "tasksampler.framework=0",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)

def main_wrapper_geo_7b_topk_noinit(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'geo3k'
    exp_name = 'verl-7b-geo-topk-noinit'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/Qwen2.5-VL-7B-Instruct"
    # MODEL_PATH = f"Qwen/Qwen2.5-VL-7B-Instruct"
    if not os.path.exists(MODEL_PATH):
        os.system("huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct")
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/geo3k/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/geo3k/test.parquet"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/geo3k.py --local_dir='data/geo3k'")
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = False
    filter_groups_metric = 'acc'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = False
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 2}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
                "actor_rollout_ref.actor.grad_clip=1.0",
                f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
                "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=10",
                "trainer.save_freq=10",
                "trainer.total_epochs=400",
                "trainer.total_training_steps=120",
                "trainer.max_actor_ckpt_to_keep=5",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=16",
                "tasksampler.framework=4",
                "tasksampler.bandit_sample_strategy='topk'",
                "tasksampler.bandit_init=False",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)


def main_wrapper_geo_7b_dapo(is_debug=False):
    from recipe.ours.main_our_remote import main
    if is_debug:
        nnodes = 1
        n_gpus_per_node=1
    else:
        nnodes = 2
        n_gpus_per_node=4
    # 设置训练参数
    project_name = 'geo3k'
    exp_name = 'verl-7b-geo-dapo'
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    RAY_DATA_HOME = f"{WORKING_DIR}"
    MODEL_PATH = f"{RAY_DATA_HOME}/models/Qwen2.5-VL-7B-Instruct"
    # MODEL_PATH = f"Qwen/Qwen2.5-VL-7B-Instruct"
    if not os.path.exists(MODEL_PATH):
        os.system("huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct")
    CKPTS_DIR = f"{RAY_DATA_HOME}/ckpts/{project_name}/{exp_name}"
    TRAIN_FILE = f"{RAY_DATA_HOME}/data/geo3k/train.parquet"
    TEST_FILE = f"{RAY_DATA_HOME}/data/geo3k/test.parquet"
    if not os.path.exists(TRAIN_FILE):
        os.system("python examples/data_preprocess/geo3k.py --local_dir='data/geo3k'")
    adv_estimator = 'grpo'

    use_kl_in_reward = False
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = 0.0

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024
    max_response_length = 1024
    enable_overlong_buffer = False
    overlong_buffer_len = 1024 * 4
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"

    enable_filter_groups = True
    filter_groups_metric = 'score'
    max_num_gen_batches = 10

    train_prompt_bsz = 256
    gen_prompt_bsz = train_prompt_bsz
    train_prompt_mini_bsz = 128
    n_resp_per_prompt = 8

    # Algorithm
    temperature = 1.0
    val_temperature = 0.6
    top_p = 1.0
    top_k = -1  # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz = False
    infer_micro_batch_size=256
    train_micro_batch_size=128
    offload = False

    with initialize(config_path="recipe/ours/config", version_base=None):
        cfg = compose(
            config_name="our_trainer",
            overrides=[
                f"data.train_files='{TRAIN_FILE}'",
                f"data.val_files='{TEST_FILE}'",
                "data.prompt_key=prompt",
                "data.truncation='left'",
                f"data.max_prompt_length={max_prompt_length}",
                f"data.max_response_length={max_response_length}",
                f"data.train_batch_size={train_prompt_bsz}",
                f"data.gen_batch_size={gen_prompt_bsz}",
                f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
                f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
                f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
                f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
                "actor_rollout_ref.actor.clip_ratio_c=10.0",
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
                f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 2}",
                f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={(max_prompt_length + max_response_length) * 12}",
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.actor.optim.weight_decay=0.1",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size={train_micro_batch_size}",
                f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
                "actor_rollout_ref.actor.entropy_coeff=0.001",
                "actor_rollout_ref.actor.grad_clip=1.0",
                f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
                "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
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
                "reward_model.reward_manager=naive",
                f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
                f"reward_model.overlong_buffer.len={overlong_buffer_len}",
                f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
                "trainer.logger=['console','wandb']",
                f"trainer.project_name={project_name}",
                f"trainer.experiment_name={exp_name}",
                f"trainer.n_gpus_per_node={n_gpus_per_node}",
                f"trainer.nnodes={nnodes}",
                "trainer.val_before_train=False",
                "trainer.test_freq=10",
                "trainer.save_freq=10",
                "trainer.max_actor_ckpt_to_keep=5",
                "trainer.total_epochs=400",
                "trainer.total_training_steps=120",
                f"trainer.default_local_dir={CKPTS_DIR}",
                "trainer.resume_mode=disable",
                "tasksampler.ts_ratio=1",
                "tasksampler.framework=0",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        return main(cfg)

if __name__ == "__main__":
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    debug = os.environ.get("DEBUG", "0")
    print(f"slurm_job_id: {slurm_job_id}")
    slurm_job_id = str(slurm_job_id)

    if debug=="1":
        print("debug mode")
        main_wrapper_geo_7b_uniform(is_debug=True)
        exit()

    if slurm_job_id == "5163046":
        main_wrapper_deepscaler_uniform()
    elif slurm_job_id == "5163047":
        main_wrapper_deepscaler_topk()
    elif slurm_job_id == "5163048":
        main_wrapper_deepscaler_ps()
    elif slurm_job_id == "5166124":
        main_wrapper_deepscaler_ps_noinit() 
    elif slurm_job_id == "5166125":
        main_wrapper_deepscaler_topk_noinit()
    #########################################################
    elif slurm_job_id == "5170243":
        main_wrapper_deepscaler_ps_upperdecay() 
    elif slurm_job_id == "5170333":#will stop?
        main_wrapper_deepscaler_srpo() 
    elif slurm_job_id == "5170327":
        main_wrapper_math_uniform() 
    elif slurm_job_id == "5170328":
        main_wrapper_math_topk() 
    elif slurm_job_id == "5170329":
        main_wrapper_math_ps() 
    elif slurm_job_id == "5170330":
        main_wrapper_math_topk_noinit() 
    elif slurm_job_id == "5170331":
        main_wrapper_math_ps_noinit() 
    elif slurm_job_id == "5175950":
        main_wrapper_math_dapo()
    elif slurm_job_id == "5175914xxx":
        main_wrapper_math_dapo_thres()
    elif slurm_job_id == "5175915":
        main_wrapper_math_7b_uniform()
    elif slurm_job_id == "5176608":
        main_wrapper_math_7b_topk_noinit()
    elif slurm_job_id == "5175917":
        main_wrapper_math_7b_dapo()
    elif slurm_job_id == "5181740":
        main_wrapper_math_topk_offline()
    elif slurm_job_id == "5180739":
        main_wrapper_math_7b_topk_noinit_new()
    elif slurm_job_id == "5191651":
        main_wrapper_geo_7b_uniform(is_debug=True)
    elif slurm_job_id == "5190732":
        main_wrapper_geo_7b_topk_noinit()
    elif slurm_job_id == "5190733":
        main_wrapper_geo_7b_dapo()
    else:
        raise ValueError(f"Invalid slurm job id: {slurm_job_id}")
