#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=verl
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=tao.hu@lmu.de
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --exclude=mcml-hgx-h100-018,mcml-hgx-h100-019

# export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=e07f72e77a1e06646e8091c6fd2e8b1c9a197a17


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "nodes_array: ${nodes_array[*]}"
# Head 节点名称
head_node=${nodes_array[0]}
port=26379
ip_head=$head_node:$port
export ip_head
# Add this before starting Ray
export GLOO_SOCKET_IFNAME="ibo8.8005"  # or your specific network interface, https://github.com/volcengine/verl/issues/967

echo "Head Node IP with port: $ip_head"




# 启动 Head 节点上的 Ray
echo "Starting HEAD node: $head_node"
srun --nodes=1 --ntasks-per-node=4 --container-image=/dss/mcmlscratch/0E/di35zis/lab/verl/verlraw2.sqsh --container-mounts=/dss/mcmlscratch/0E/di35zis/lab/verl:/workspace/verl  -w "$head_node" \
    ray start --head --node-ip-address="$head_node" --port=$port \
     --num-gpus=4 --block &

sleep 10  # 等待 Head 启动

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks-per-node=4  --container-mounts=/dss/mcmlscratch/0E/di35zis/lab/verl:/workspace/verl  --container-image=/dss/mcmlscratch/0E/di35zis/lab/verl/verlraw2.sqsh -w "$node_i" \
            ray start --address "$ip_head" --num-gpus 4 --block &
    sleep 5
done


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="hfmodels/DeepSeek-R1-Distill-Qwen-1.5B"
fi

MODEL_PATH=$(readlink -f "$MODEL_PATH")




# 启动训练（仍然在 Head 节点）
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks-per-node=4 --container-image=/dss/dsshome1/0E/di35zis/lab/verl/verlraw2.sqsh   --container-mounts=/dss/dsshome1/0E/di35zis/lab/verl:/workspace/verl -w "$head_node" \
bash -c "mkdir -p /workspace/verl/.cache/huggingface  && mkdir -p /workspace/verl/.cache/flashinfer && cd /workspace/verl &&  export HF_HOME=/workspace/verl/.cache/huggingface &&              export TRANSFORMERS_CACHE=/workspace/verl/.cache/huggingface &&              export HF_DATASETS_CACHE=/workspace/verl/.cache/huggingface &&              export FLASHINFER_WORKSPACE_DIR=/workspace/verl/.cache/flashinfer && export HOME=/workspace/verl/  &&   python train_wrapper.py"