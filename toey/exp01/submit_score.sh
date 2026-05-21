#!/bin/bash
#SBATCH --job-name=exp01-score
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/logs/exp01_score_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/logs/exp01_score_%j.err

set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$PROJECT/venv/bin/activate"

export HF_HOME="$PROJECT/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 "$PROJECT/ua047/textsum/eval_train/score.py" \
    "$PROJECT/toey/exp01/result/submission.csv"
