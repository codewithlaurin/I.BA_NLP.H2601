#!/bin/bash
# Stages 1-3 on Clariden: one W&B sweep agent per GPU on one GH200 node (4 GPUs). Clariden allocates
# and charges whole nodes, so all four GPUs are used. Each agent pulls grid points until its sweep has
# none left, then exits. With several sweep IDs, the GPUs are shared out in turn (2 sweeps: 2 GPUs each).
#
# Usage (from the repository root):
#   sbatch --account=<project> docs/project2/clariden_sweep.sbatch <SWEEP_ID> [<SWEEP_ID> ...]
#SBATCH --job-name=p2-sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=outputs/project2/slurm/%x-%j.out

set -euo pipefail
sweep_ids=("$@")
gpus=4
if [[ ${#sweep_ids[@]} -eq 0 ]]; then
    echo "Usage: sbatch --account=<project> $0 <SWEEP_ID> [<SWEEP_ID> ...]" >&2
    exit 1
fi

for gpu in $(seq 0 $((gpus - 1))); do
    sweep_id=${sweep_ids[$((gpu % ${#sweep_ids[@]}))]}
    echo "GPU slot ${gpu}: sweep ${sweep_id}"
    # Slurm sets CUDA_VISIBLE_DEVICES per step, so each agent sees exactly one GPU.
    srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --mem=100G \
        --output="outputs/project2/slurm/%x-%j-gpu${gpu}.out" \
        uv run wandb agent "laurin-uni-workspace/nlp-transformers/${sweep_id}" &
done
wait
