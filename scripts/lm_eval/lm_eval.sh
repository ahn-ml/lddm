checkpoint_path=/data/mingyu/loopholing_ckpts/owt_memory_1M.ckpt
job_name="lddm_m_no_padding_topk"
task=lambada_openai
CUDA_VISIBLE_DEVICES=0 python lm_eval_harness.py \
  --batch_size 64 \
  --tasks $task \
  --model mgm \
  --model_args checkpoint_path=${checkpoint_path} \
  --output_path ./harness_results/${job_name}/${task}.json
echo "Finished evaluating task: $task"