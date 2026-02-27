# Loopholing Discrete Diffusion Models

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/) [![arXiv](https://img.shields.io/badge/arXiv-2510.19304-b31b1b)](https://arxiv.org/abs/2510.19304) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://sites.google.com/view/lddms/home)

This repository provides the official PyTorch implementation for **Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall**.

Our work demonstrates how maintaining latent embeddings that preserve prediction information in discrete diffusion enables significant performance improvements in text generation tasks.

<img src="assets/lddm_large_resolution.gif" alt="LDDM demo">

## Code Structure

```
├── main.py              # Entry point (train / ppl_eval / sample_eval)
├── algo.py              # All algorithm implementations
├── trainer_base.py      # Base classes (Diffusion, AbsorbingState, UniformState)
├── models/
│   └── dit.py           # Diffusion Transformer backbone
├── dataloader.py        # Data loading and tokenization
├── metrics.py           # NLL, Perplexity, Generative Perplexity, etc.
├── lm_eval_harness.py   # Downstream task evaluation (LM Eval Harness)
├── configs/             # Hydra configs (algo, data, model, noise, strategy)
└── scripts/             # Training and evaluation shell scripts
```

**Algorithm hierarchy:**
- `MDLM` → `LDDM_M` (masked diffusion + loophole)
- `UDLM` → `LDDM_U` (uniform diffusion + loophole)
- Baselines: `SEDD`, `D3PM`, `AR`

## Pretrained Checkpoints

Pretrained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1oiLQw8fe6UJ64OY09u9WZ8_xSckJ6Qb8?usp=sharing).

## Installation

```bash
conda create -n lddm python=3.12
conda activate lddm
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

## Training

Training scripts are available in `scripts/train/`:
- **LM1B**: `train_lm1b_lddm_m.sh`, `train_lm1b_lddm_u.sh`
- **OpenWebText**: `train_owt_lddm_m.sh`, `train_owt_lddm_u.sh`

Example:
```bash
bash scripts/train/train_lm1b_lddm_m.sh
```

## Evaluation

Evaluation scripts are in `scripts/eval/`:
- **Perplexity**: `eval_lm1b_lddm_m.sh`, `eval_owt_lddm_m.sh`
- **Generation**: `gen_owt_lddm_m.sh`

Downstream task scripts are in `scripts/lm_eval/`:
- **Downstream**: `lm_eval.sh`

Example:
```bash
bash scripts/eval/eval_owt_lddm_m.sh
```

## Baselines

Baseline implementations (SEDD, MDLM, UDLM, D3PM, AR) are also included:
- Training: `scripts/train/train_*_{sedd,mdlm,udlm}.sh`
- Evaluation: `scripts/eval/eval_*_{sedd,mdlm,udlm}.sh`

## Citation

```bibtex
@misc{jo2025loopholingdiscretediffusiondeterministic,
      title={Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall}, 
      author={Mingyu Jo and Jaesik Yoon and Justin Deschenaux and Caglar Gulcehre and Sungjin Ahn},
      year={2025},
      eprint={2510.19304},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.19304}, 
}
```

## Acknowledgements

This repository is based on [DUO](https://github.com/s-sahoo/duo).
