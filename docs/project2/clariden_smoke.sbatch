sbatch --account=g239 docs/project2/clariden_smoke.sbatch ekp6r2o7

cat > docs/project2/clariden_smoke.sbatch <<'EOF'
#!/bin/bash
# Stage 0 on Clariden: environment check, then the P2-Smoke sweep run, interrupted after epoch 1
# and resumed, so the sweep path, run naming, resuming and checkpoint restore are all exercised.
#
# Usage (from the repository root):
#   uv run wandb sweep docs/project2/sweep_0_smoke.yaml        # prints the sweep ID
#   sbatch --account=<project> docs/project2/clariden_smoke.sbatch <SWEEP_ID>
#SBATCH --job-name=p2-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=outputs/project2/slurm/%x-%j.out

set -euo pipefail
sweep_id=$1

echo "== Environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run python -c "
import torch
assert torch.cuda.is_available(), 'PyTorch does not see a GPU'
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))
"

echo "== Sweep run until epoch 1 is checkpointed"
started_marker=$(mktemp)
uv run wandb agent --count 1 "laurin-uni-workspace/nlp-transformers/${sweep_id}" &
agent_pid=$!
run_dir=""
# last.pt is written atomically at the end of an epoch, so its existence means epoch 1 is resumable.
until [[ -n "$run_dir" && -e "$run_dir/last.pt" ]]; do
    sleep 10
    kill -0 "$agent_pid"  # fails (and ends the job) if the agent exited before epoch 1
    run_dir=$(find outputs/project2 -maxdepth 1 -name 'pretrained_linear_lr1e-5_seed42_*' \
        -newer "$started_marker" | head -1)
done
echo "Interrupting ${run_dir} after epoch 1"
pkill -INT -f train_transformer.py || true
wait "$agent_pid" || true

echo "== Resume the same run"
uv run python train_transformer.py --initialization pretrained --scheduler linear \
    --learning-rate 1e-5 --epochs 2 --patience 10 --seed 42 --batch-size auto \
    --device cuda --wandb-group P2-Smoke --resume "${run_dir}/last.pt"

echo "== Checks"
cat "${run_dir}/summary.json"
grep -E '"batch_size"|"gpu"' "${run_dir}/config.json"
uv run python -c "
import json, sys
for line in open(sys.argv[1]):
    m = json.loads(line)
    print('epoch', m['epoch'], '| elapsed', round(m['elapsed_seconds']), 's | lr', m['learning_rate'],
          '| validation accuracy', m['validation']['accuracy'])
" "${run_dir}/metrics.jsonl"
if [[ -e "${run_dir}/last.pt" ]]; then echo "FAIL: last.pt still exists"; else echo "OK: last.pt removed"; fi

EOF
