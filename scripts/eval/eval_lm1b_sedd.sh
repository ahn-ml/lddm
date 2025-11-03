checkpoint_path=/data/mingyu/loopholing_ckpts/lm1b_sedd_1M.ckpt

export HYDRA_FULL_ERROR=1

datasets=("ag_news"
          "scientific_papers_pubmed"
          "scientific_papers_arxiv"
          "lambada"
          "wikitext2"
          "ptb"
          "lm1b")
for data in "${datasets[@]}"; do
  echo "$data"
  CUDA_VISIBLE_DEVICES=0,1 python3 -u -m main \
    mode=ppl_eval \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.eval_global_batch_size=512 \
    data="$data" \
    data.insert_valid_eos=True \
    data.wrap=False \
    data.tokenizer_name_or_path="bert-base-uncased" \
    model=small \
    algo=sedd \
    sampling.predictor=analytic \
    model.length=128 \
    eval.checkpoint_path=$checkpoint_path \
    sampling.num_sample_batches=0 \
    +wandb.offline=true
done