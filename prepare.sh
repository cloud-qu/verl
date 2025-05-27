
#!/usr/bin/env bash
set -euo pipefail

export PS1=""
source "$(conda info --base)/etc/profile.d/conda.sh"
#conda create -n rllm python=3.10 -y
#conda activate rllm
conda install nvidia/label/cuda-12.1.0::cuda-tools
conda install -c nvidia cuda-toolkit

USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install --no-deps -e .

cd  recipe/deepscaler
python deepscaler_dataset.py --local_dir='processed_data'
cd ../..


# download model
mkdir -p hfmodels
pip install -U huggingface_hub
#export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B

# wandb
pip install wandb
export WANDB_BASE_URL="http://182.18.90.106:8777"
export WANDB_API_KEY="e07f72e77a1e06646e8091c6fd2e8b1c9a197a17"
wandb login --relogin "$WANDB_API_KEY"
