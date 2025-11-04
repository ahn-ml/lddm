#!/usr/bin/env bash

# task list
tasks=("lambada_openai" "hellaswag" "arc_easy" "arc_challenge" "mathqa" "piqa" "winogrande")
checkpoint_path="/data/mingyu/owt_memory_1M.ckpt"
job_name="lddm_m_no_padding_topk"

# current number of GPUs available
NUM_GPUS=4
declare -a available_gpus
for ((i=0; i<NUM_GPUS; i++)); do
  available_gpus[i]=$i
done

# array to hold PIDs of running jobs on each GPU
declare -a gpu_pids
for ((i=0; i<NUM_GPUS; i++)); do
  gpu_pids[i]=""
done

# job index
job_idx=0
total_jobs=${#tasks[@]}

echo "Total tasks: $total_jobs"
echo "Tasks: ${tasks[@]}"
echo "Available GPUs: ${available_gpus[@]} (Total: $NUM_GPUS)"

# main loop to assign tasks to GPUs
while [ $job_idx -lt $total_jobs ] || [ $(jobs -r | wc -l) -gt 0 ]; do
  # assign tasks to available GPUs
  for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    # check if there's a running job on this GPU
    if [ -n "${gpu_pids[gpu_idx]}" ]; then
      # check if the process is still running
      if ! kill -0 ${gpu_pids[gpu_idx]} 2>/dev/null; then
        echo "GPU ${available_gpus[gpu_idx]}: Job finished (PID ${gpu_pids[gpu_idx]})"
        gpu_pids[gpu_idx]=""
      fi
    fi
    
    # if no job running on this GPU, assign a new task
    if [ -z "${gpu_pids[gpu_idx]}" ] && [ $job_idx -lt $total_jobs ]; then
      task=${tasks[job_idx]}
      gpu_id=${available_gpus[gpu_idx]}
      echo ">>> Starting job $((job_idx+1))/$total_jobs: task=${task} on GPU ${gpu_id}"
      
      CUDA_VISIBLE_DEVICES=${gpu_id} \
      python lm_eval_harness.py \
        --batch_size 64 \
        --tasks ${task} \
        --model mgm \
        --model_args checkpoint_path=${checkpoint_path} \
        --output_path ./harness_results/${job_name}/${task}.json \
        > log_${task}_gpu${gpu_id}.log 2>&1 &
      
      gpu_pids[gpu_idx]=$!
      echo "GPU ${gpu_id}: Started job with PID ${gpu_pids[gpu_idx]} for task ${task}"
      job_idx=$((job_idx+1))
    fi
  done
  
  # wait before next check
  sleep 10
done

echo "All LM evaluations completed!"
echo "Check log files: log_*_gpu*.log"
echo "Check results in: ./harness_results/mdlm/"