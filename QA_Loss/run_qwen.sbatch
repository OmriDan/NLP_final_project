#!/bin/bash
#SBATCH --job-name=qwen_loss
#SBATCH --partition=studentkillable        # your GPU partition
#SBATCH --account=gpu-students             # your billing account
#SBATCH --gres=gpu:1                       # request 1 GPU
#SBATCH --cpus-per-task=4                  # adjust as needed
#SBATCH --mem=32G                          # RAM per node
#SBATCH --time=08:00:00                  # max runtime: 2 days
#SBATCH --output=/home/joberant/NLP_2425a/idandrori/logs/qwen-%j.out
#SBATCH --error=/home/joberant/NLP_2425a/idandrori/logs/qwen-%j.err

# enable PyTorch’s expandable‑segments allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# # Initialize the module system (compute nodes start a clean login shell)
source /etc/profile.d/modules.sh

# # Load CUDA so PyTorch can see the GPU
module load cuda/12.0

# Go to your project and activate the virtual environment you built on NetApp
cd ~/NLP_proj/QA_Loss
source venv-qa_loss/bin/activate

# (Optional) ensure HF and cache dirs point at your NetApp area
export HF_HOME=/home/joberant/NLP_2425a/idandrori/hf_cache
export XDG_CACHE_HOME=/home/joberant/NLP_2425a/idandrori/cache

# Run your script
python -u calculate_qa_loss_sequence_qwen.py