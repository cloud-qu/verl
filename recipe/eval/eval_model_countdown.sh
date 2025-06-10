set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_P2P_DISABLE=1

# Default values
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("test")
OUTPUT_DIR="$MODEL_PATH/eval_results/"  # Add default output directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done
# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# 如果 ${MODEL_PATH}_hf 已存在且包含 .safetensors，优先使用；否则若原始 MODEL_PATH 不含 .safetensors，则合并生成 hf 目录
if [[ -d "${MODEL_PATH}_hf" && $(ls "${MODEL_PATH}_hf"/*.safetensors 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "Found .safetensors in ${MODEL_PATH}_hf, use it directly"
    MODEL_PATH="${MODEL_PATH}_hf"
elif ! ls "${MODEL_PATH}"/*.safetensors &>/dev/null; then
    echo "No .safetensors found in ${MODEL_PATH}, merging to hf…"
    python scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "${MODEL_PATH}" \
        --target_dir "${MODEL_PATH}_hf"
    MODEL_PATH="${MODEL_PATH}_hf"
    echo "Switched MODEL_PATH to ${MODEL_PATH}"
fi


project_name='countdown_eval'
IFS='/' read -ra parts <<< "$MODEL_PATH"
len=${#parts[@]}

experiment_name=${parts[$((len-3))]}/${parts[$((len-2))]}

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m recipe.eval.main_eval \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=$HOME/verl/data/countdown3to4/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        data.n_samples=16 \
        data.batch_size=512 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=8192 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1 \
        wandb.project_name=${project_name} \
        wandb.experiment_name=${experiment_name}/${DATA_TYPE}
done
