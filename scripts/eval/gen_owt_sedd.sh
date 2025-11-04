checkpoint_path=/data/mingyu/loopholing_ckpts/owt_sedd_1M.ckpt

CUDA_VISIBLE_DEVICES=1 python main.py \
  mode=sample_eval \
  loader.eval_batch_size=4 \
  data=openwebtext-split \
  model=small \
  model.length=1024 \
  algo=sedd \
  sampling.predictor=analytic \
  eval.checkpoint_path=$checkpoint_path \
  sampling.steps=1024 \
  sampling.num_sample_batches=2 \
  sampling.predictor=analytic \
  sampling.use_float64=True \
  +wandb.offline=true