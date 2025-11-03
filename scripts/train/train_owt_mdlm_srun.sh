#!/bin/bash
#SBATCH -J mdlm-lm1b                   # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 2                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu_jobs          # Request partition (DO NOT EDIT)
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

srun python -u -m main \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  loader.num_workers=8 \
  data=openwebtext-split \
  wandb.name=mdlm_memory \
  algo=mdlm \
  model=small \
  model.length=1024