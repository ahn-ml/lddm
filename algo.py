import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step):
    del current_accumulation_step
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  def generate_samples(self, num_samples, kl_step=20, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    for i in range(num_pred_tokens):
      output = self.backbone(x[:, :i + 1], None)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    assert self.config.sampling.predictor == 'ancestral' # for safety

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0

    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x 
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(
      model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_index).to(
      model_output.dtype)[:, :, None]
    model_output = (
      masked_score * masked_indices
      + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class LDDM_M(MDLM):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.algo.loophole == True
    assert self.config.algo.backbone == 'dit' # didn't tested at others
    assert self.config.sampling.predictor == 'ancestral'


class D3PMAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.noise.type == 'log-linear'
    assert self.parameterization == 'mean'

  def _process_model_output(self, model_output, xt, sigma):
    del xt 
    del sigma
    if self.subs_masking:
      model_output[:, :, self.mask_index] += self.neg_infinity
    return model_output.log_softmax(dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del dalpha_t
    assert not low_var
    dt = 1 / self.T
    t = 1 - alpha_t  # Only valid for log-linear schedule.
    t = t.clamp(0., 1.0 - 1e-4)
    alpha_t = alpha_t + torch.zeros_like(xt)
    alpha_s = t - dt + torch.zeros_like(xt)
    assert alpha_s.shape == xt.shape
    assert alpha_t.shape == xt.shape
    log_x_theta_at_x0 = torch.gather(
      log_x_theta, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(
      alpha_s * x_theta_at_m / (t - dt) + 1)
    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
    return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.predictor == 'analytic'
    assert self.config.algo.loophole == False

  def _get_score(self, x, sigma):
    return self.forward(x, sigma).exp()

  def _process_model_output(self, model_output, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(model_output.dtype)
    # logits shape
    # (batch_size, context_length, vocab_size)
    model_output = (model_output
                    - esigm1_log[:, None, None]
                    - np.log(model_output.shape[-1] - 1))
    # The below scatter operation sets the log score
    # for the input word to 0.
    model_output = torch.scatter(
      model_output, -1, xt[..., None],
      torch.zeros_like(model_output[..., :1]))
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    """Computes the SEDD loss for the Absorbing State Diffusion.

    Args:
      log_x_theta: float torch.Tensor with shape (batch_size,
          context_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          context_length), input.
      x0: int torch.Tensor with shape (batch_size,
          context_length), input.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
          time derivative of signal level.
      low_var: bool, low variance loss during training.
    
    Returns:
      loss with shape (batch_size, context_length).
    """
    assert not low_var
    masked_indices = xt == self.mask_index
    sigma = self._sigma_from_alphat(alpha_t)
    dsigma = - dalpha_t / alpha_t

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_x_theta[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_x_theta[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return dsigma * entropy


class UDLM(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    return model_output.log_softmax(dim=-1)

  def _compute_posterior(self, x, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot),
        or model's predicted `x_theta` of shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    if self.config.sampling.use_float64:
      x = x.to(torch.float64)
    if alpha_s.ndim == 2:
      alpha_s = alpha_s.unsqueeze(-1)
    if alpha_t.ndim == 2:
      alpha_t = alpha_t.unsqueeze(-1)
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size).to(
      self.dtype).to(self.device)
    return (
      (alpha_t * self.vocab_size * x * xt_one_hot + (
        alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
          1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
            alpha_t * self.vocab_size * torch.gather(
              x, -1, xt[..., None]) + (1 - alpha_t)))

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    term1 = self.vocab_size * (1 / xbar_xt
                                - 1 / xbar_theta_xt)
    
    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False, prev_latent=None):
    del p_x0
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    sigma_t = self._sigma_from_alphat(alpha_t)
    assert alpha_t.ndim == 2
    
    pred_x = self.forward(x, sigma_t)
    if self.config.sampling.use_float64:
      pred_x = pred_x.to(torch.float64)
    q_xs = self._compute_posterior(
      x=pred_x.exp(),
      xt=x,
      alpha_s=alpha_s,
      alpha_t=alpha_t)
    if self.p_nucleus < 1:
      q_xs = utils.top_k_top_p_filtering(
        q_xs.log(), top_p=self.p_nucleus)
    return None, trainer_base.sample_categorical(q_xs), pred_x


class LDDM_U(UDLM):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.noise.type == 'log-linear'
    assert self.parameterization == 'mean'
    assert self.config.algo.loophole == True
    assert self.config.algo.backbone == 'dit' # didn't tested at others
    assert self.config.sampling.predictor == 'ancestral'


  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False, prev_latent=None):
    del p_x0
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    sigma_t = self._sigma_from_alphat(alpha_t)
    assert alpha_t.ndim == 2
    
    pred_x, latent = self.forward(x, sigma_t, prev_latent=prev_latent)
    if self.config.sampling.use_float64:
      pred_x = pred_x.to(torch.float64)
    q_xs = self._compute_posterior(
      x=pred_x.exp(),
      xt=x,
      alpha_s=alpha_s,
      alpha_t=alpha_t)
    if self.p_nucleus < 1:
      q_xs = utils.top_k_top_p_filtering(
        q_xs.log(), top_p=self.p_nucleus)
    return None, trainer_base.sample_categorical(q_xs), latent, pred_x
