checkpoint_path=/data/mingyu/loopholing_ckpts/owt_memory_1M.ckpt

export HYDRA_FULL_ERROR=1

datasets=("ag_news"
          "scientific_papers_pubmed"
          "scientific_papers_arxiv"
          "lambada"
          "wikitext2"
          "ptb"
          "lm1b-gpt2"
          "openwebtext-split")
for data in "${datasets[@]}"; do
  echo "$data"
  CUDA_VISIBLE_DEVICES=0,1 python3 -u -m main \
    mode=ppl_eval \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.eval_global_batch_size=512 \
    data="$data" \
    data.insert_valid_eos=True \
    model=small \
    algo=lddm_m \
    algo.self_cond_rate=1.0 \
    model.length=1024 \
    eval.checkpoint_path=$checkpoint_path \
    sampling.num_sample_batches=0 \
    +wandb.offline=true
done