import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# NOTE: PyTorch>=2.6 defaults to weights_only=True in torch.load.
# This project uses legacy Lightning checkpoints that include non-tensor
# hyperparameters (e.g., OmegaConf/Tokenizer objects), so force full load.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def self_bleu(sentences, max_n=4):
    smoothie = SmoothingFunction().method1
    weights = tuple([1/max_n]*max_n)  # e.g. (0.25,0.25,0.25,0.25) for 4-gram

    scores = []
    for i, sent in tqdm(enumerate(sentences), desc="Computing Self-BLEU", total=len(sentences)):
        cand = word_tokenize(sent.lower())
        refs = [
            word_tokenize(other.lower())
            for j, other in enumerate(sentences) 
            if j != i
        ]
        score = sentence_bleu(
            refs, cand,
            weights=weights,
            smoothing_function=smoothie
        )
        scores.append(score)

    return np.mean(scores)

def save_kl_divergence(total_kl, kl_step, img_path="kl_divergence.png", log_path="kl_divergence.json"):
    plt.clf() 
    plt.plot([kl_step * i for i in range(len(total_kl))],total_kl)
    plt.title("KL divergence over steps")
    plt.yscale("log")
    plt.ylabel("KL divergence (log scale)")
    plt.xlabel("step")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    with open(log_path, 'w') as f:
      json.dump(total_kl, f)

def save_entropy(total_entropy, img_path="entropy.png", log_path="entropy.json"):
    plt.clf()
    plt.plot([i for i in range(len(total_entropy))],total_entropy)
    plt.title("Entropy of tokens")
    plt.yscale("log")
    plt.ylabel("Entropy (log scale)")
    plt.xlabel("step")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    with open(log_path, 'w') as f:
      json.dump(total_entropy, f)

def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_samples = []
  total_kl = []
  total_entropy = []
  kl_step = config.eval.kl_step
  for _ in tqdm(range(config.sampling.num_sample_batches), desc="Generating samples"):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples, kl_steps, entropy_steps = model.restore_model_and_sample(
        num_steps=config.sampling.steps, kl_step=kl_step)
      total_kl += kl_steps
      total_entropy += entropy_steps
      model.metrics.record_entropy(samples)
      text_samples = model.tokenizer.batch_decode(samples)
      model.metrics.record_generative_perplexity(
        text_samples, config.model.length, model.device)
      all_samples.extend(list(text_samples))
  generative_ppl = 0.
  entropy = 0.
  if not config.sampling.semi_ar:
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)
    self_bleu_score = self_bleu(all_samples, max_n=4)
    print("Self-BLEU score:", self_bleu_score)
  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'generative_ppl': generative_ppl,
               'entropy': entropy,
                'self_bleu': self_bleu_score,
               'generated_seqs': all_samples}, f, indent=4)

  if config.eval.kl_divergence_eval:
    total_kl = torch.stack(total_kl, dim=0).mean(dim=0).cpu().numpy().tolist()
    total_entropy = torch.stack(total_entropy, dim=0).mean(dim=0).cpu().numpy().tolist()
    save_kl_divergence(total_kl, kl_step)
    save_entropy(total_entropy)
  print('Samples saved at:', samples_path)

def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config)
  else:
    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'lddm_m':
    diffusion_model = algo.LDDM_M
  elif config.algo.name == 'lddm_u':
    diffusion_model = algo.LDDM_U
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'udlm':
    diffusion_model = algo.UDLM
  elif config.algo.name == 'd3pm':
    diffusion_model = algo.D3PMAbsorb
  elif config.algo.name == 'sedd':
    diffusion_model = algo.SEDDAbsorb
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()