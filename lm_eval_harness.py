import torch

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import algo
import dataloader
import yaml
import os
from omegaconf import OmegaConf


def requests_to_dataset(config, requests, tokenizer, num_proc):
  def _tokenize(e):
    eos_idx = tokenizer.eos_token_id
    bos_idx = tokenizer.bos_token_id
    prefix_tokens = tokenizer(e['prefix'], 
                              return_attention_mask=False, 
                              add_special_tokens=False
                              )['input_ids']
    target_tokens = tokenizer(e['target'], 
                              return_attention_mask=False, 
                              add_special_tokens=False
                              )['input_ids']
    prefix_tokens = [bos_idx] + prefix_tokens
    target_tokens = target_tokens + [eos_idx]
    
    return {
        'prefix_text': e['prefix'],
        'target_text': e['target'],
        'prefix': prefix_tokens,
        'target': target_tokens,
    }
  ds = []
  ds = [{'prefix': req.args[0], 'target': req.args[1]} 
        for req in requests]
  ds = Dataset.from_list(ds)
  ds = ds.map(_tokenize, num_proc=num_proc)
  ds = ds.with_format('torch')
  seq_lenths = [len(x['prefix']) + len(x['target']) 
                for x in ds]
  if max(seq_lenths) > config.model.length:
    print('ERROR: TEXT TOO LONG!!!!')
    breakpoint()

  return ds


def _eval_suffix_nll_generators(config, module, prefix, suffix,
                                batch_size, num_samples):
  device = module.device
  assert num_samples % batch_size == 0
  full_sentence = torch.cat([prefix, suffix], dim=-1
                  ).repeat(batch_size, 1).to(module.device)
  all_ts = module._sample_t(num_samples, accum_step=None)
  for idx in range(0, num_samples, batch_size):
    t = all_ts[idx:idx+batch_size].unsqueeze(-1)
    dalpha_t, alpha_t = module.noise(t)
    alpha_t = alpha_t.to(device)
    sigma = module._sigma_from_alphat(alpha_t)
    x0 = full_sentence.to(device)
    if config.algo.name == 'mdlm' or config.algo.name == 'lddm_m':
      xt = module.q_xt(full_sentence, alpha_t).to(device)
      group_idxs = None
    else:
      raise ValueError(config.algo.name)
    yield xt, x0, group_idxs, sigma, alpha_t, dalpha_t
    

def eval_suffix_nll(config, module, prefix, suffix, batch_size, 
                    num_samples, padding=False, loophole=False):
  all_losses = []
  generator =  _eval_suffix_nll_generators(config, module, 
                  prefix, suffix, batch_size, num_samples)
  for xt, x0, group_idxs, sigma, alpha_t, dalpha_t in generator:
    cond = torch.zeros_like(sigma)  # No time conditioning
    if group_idxs is not None:
      log_x_theta = module(xt, cond, group_idxs=group_idxs)
    else:
      if padding:
        seq_length = xt.shape[1]
        if seq_length < config.model.length:
          xt = torch.cat([xt, torch.ones(
              (xt.shape[0], config.model.length - seq_length),
              device=xt.device) * module.mask_index], dim=-1)
          xt = xt.long()
      if loophole:
        log_x_theta, latent = module(xt, cond, prev_latent=None)
        log_x_theta, latent = module(xt, cond, prev_latent=latent)
      else:
        log_x_theta = module(xt, cond)
      if padding:
        log_x_theta = log_x_theta[:, :seq_length]
    token_nll = module.nll_per_token(log_x_theta, xt, x0, 
                                     alpha_t, dalpha_t)
    if group_idxs is not None:
      # Assume group 1 is masked
      token_nll = token_nll * group_idxs
    all_losses.append(float(token_nll.mean()))
  return float(np.mean(all_losses))


@register_model("mgm")
class MGMEvalWrapper(LM):
  def __init__(self, pretrained="NONE", max_length=1024,
               num_mc_samples=1024, batch_size=64, device="cuda",
               checkpoint_path=None, num_proc=8):
    super().__init__()
    ckpt = torch.load(checkpoint_path, map_location='cpu', 
                      weights_only=False)
    config = ckpt['hyper_parameters']['config']

    # if there is no 'algo' attribute in config, you should choose one manually
    if not hasattr(config, 'algo'):
      algo_name = 'lddm_m'  # 'lddm_m' or 'mdlm'
      # Load algo config from yaml
      current_dir = os.path.dirname(os.path.abspath(__file__))
      algo_config_path = os.path.join(current_dir, 'configs', 'algo', f'{algo_name}.yaml')
      with open(algo_config_path, 'r') as f:
        algo_config = yaml.safe_load(f)
      
      # disable struct mode to modify config
      OmegaConf.set_struct(config, False)
      config.algo = OmegaConf.create(algo_config)
      OmegaConf.set_struct(config, True)

    self.tokenizer = dataloader.get_tokenizer(config)
    config.sampling.predictor='ancestral'
    if config.algo.name == 'mdlm':
      if not hasattr(config.algo, 'loophole'):
        OmegaConf.set_struct(config.algo, False)   # unlock struct mode
        config.algo.loophole = False
        OmegaConf.set_struct(config.algo, True)    # lock
      else:
        config.algo.loophole = False
      self.model = algo.MDLM(config, self.tokenizer)
    elif config.algo.name == 'lddm_m':
      if not hasattr(config.algo, 'loophole'):
        OmegaConf.set_struct(config.algo, False)   # unlock struct mode
        config.algo.loophole = True
        config.algo.self_cond_rate = 1.0  # for eval, always use self cond
        OmegaConf.set_struct(config.algo, True)    # lock
      else:
        config.algo.loophole = True

      self.model = algo.LDDM_M(config, self.tokenizer)
    else:
      raise ValueError(f'Implement for {config.algo.name}')
    self.config = config
    self.num_proc = num_proc
    self.num_mc_samples = num_mc_samples
    self.batch_size = int(batch_size)
    self.device = device
    
    self.padding = False
    self.model.load_state_dict(ckpt['state_dict'])
    self.model.to(device)
    self.model.eval()

  def suffix_greedy_prediction(self, prefix, target, padding=False, add_one_mask=True, loophole=False):
    if self.config.algo.name == 'mdlm' or self.config.algo.name == 'lddm_m':
      return self._suffix_greedy_prediction_mdlm(prefix, 
                                                 target, padding, add_one_mask, loophole=loophole)
    else:
      raise ValueError(self.config.algo.name)

  def _suffix_greedy_prediction_mdlm(self, prefix, target, padding=False, add_one_mask=True, loophole=False):
    mask_idx = self.model.mask_index
    eos_idx = self.tokenizer.eos_token_id
    noisy_target = [mask_idx] * (len(target) - 1) + [eos_idx]
    noisy_target = torch.tensor(noisy_target, 
                                device=self.device)
    prefix = prefix.to(self.device)
    seq = torch.concatenate([prefix, noisy_target], 
                            dim=-1).reshape(1, -1)
    sigma = torch.zeros(size=(seq.shape[0], 1), 
                        device=self.device)

    latent = None
    for i in range(len(target)-1):
      masked_idx = (seq == mask_idx)
      if add_one_mask:
        seq = torch.cat([seq[:,:-1], torch.tensor([mask_idx, eos_idx], device=self.device).reshape(1, -1)], dim=-1).long()
      if padding:
        seq_length = seq.shape[1]
        if seq_length < self.config.model.length:
          seq = torch.cat([seq, torch.ones(
              (seq.shape[0], self.config.model.length - seq_length),
              device=self.device) * self.model.mask_index], dim=-1)
          seq = seq.long()
          
      if loophole:
        if latent is None:
          logits, latent = self.model(seq, sigma, prev_latent=latent)
        logits, latent = self.model(seq, sigma, prev_latent=latent)
      else:
        logits = self.model(seq, sigma)
        
      if padding:
        seq = seq[:, :seq_length]
        logits = logits[:, :seq_length]
      if add_one_mask:
        seq = torch.cat([seq[:,:-2], torch.tensor([eos_idx], device=self.device).reshape(1, -1)], dim=-1).long()
        logits = torch.cat([logits[:,:-2], logits[:,-1].reshape(1, -1, logits.shape[-1])], dim=1)
      assert logits.shape[0] == 1
      logits = logits[masked_idx]
      x0 = torch.argmax(logits, -1)
      p = logits.exp()
      confidence = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
      _, index = torch.sort(confidence, descending=True)
      x0[index[1:]]=mask_idx
      seq[masked_idx] = x0.clone()
    
    correct = target == seq[0,len(prefix):].cpu()
    correct = correct.all()
    return bool(correct)
      
  @torch.no_grad()
  def loglikelihood(self, requests: list[Instance]) \
                                -> list[tuple[float, bool]]:
    add_one_mask= True if getattr(requests[0], "task_name", None) == "lambada_openai" else False
    dataset = requests_to_dataset(self.config, requests, 
                                  self.tokenizer, self.num_proc)
    out = []
    for elem in tqdm(dataset, 'Computing likelihood...'):
      prefix = elem['prefix']
      target = elem['target']
      ll = -eval_suffix_nll(self.config, self.model, prefix, 
                            target, self.batch_size, 
                            self.num_mc_samples, padding=self.padding, loophole=self.config.algo.loophole)
      is_target_greedy_dec = self.suffix_greedy_prediction(
        prefix, target, padding=self.padding, add_one_mask=add_one_mask, loophole=self.config.algo.loophole)
      out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
    return out

  def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
    raise NotImplementedError
  
  def generate_until(self, context, max_length, stop, 
                     **generation_kwargs):
    raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()