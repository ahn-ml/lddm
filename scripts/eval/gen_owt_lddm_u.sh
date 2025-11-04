checkpoint_path=/data/mingyu/loopholing_ckpts/owt_udlm_memory_1M.ckpt

CUDA_VISIBLE_DEVICES=0 python main.py \
  mode=sample_eval \
  loader.eval_batch_size=4 \
  data=openwebtext-split \
  model=small \
  model.length=1024 \
  algo=lddm_u \
  eval.checkpoint_path=$checkpoint_path \
  sampling.steps=1024 \
  sampling.num_sample_batches=2 \
  sampling.use_float64=True \
  eval.kl_divergence_eval=True \
  +wandb.offline=true