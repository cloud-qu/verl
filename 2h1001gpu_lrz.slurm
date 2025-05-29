#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=2h1001gpu30min_lrz
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=tao.hu@lmu.de
#SBATCH -p lrz-hgx-h100-94x4



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

source ~/.bashrc
conda activate verl2


# 启动 Head 节点上的 Ray
echo "Starting HEAD node: $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node" --port=$port \
    --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block &

sleep 10  # 等待 Head 启动

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
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
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python train_wrapper.py
