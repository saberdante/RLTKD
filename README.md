# RLTKD: Rethinking Distillation Temperature: A Reinforcement Learning Framework for Adaptive Temperature Scheduling

This is the official implementation of our paper "Rethinking Distillation Temperature: A Reinforcement Learning Framework for Adaptive Temperature Scheduling".

This project is based on the mdistiller library. Thanks to the mdistiller library (https://github.com/megvii-research/mdistiller).

### Installation

Environments:

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0

Install the package:

```
pip install -r requirements.txt
python setup.py develop
```

### Wandb as the logger

- The registeration: <https://wandb.ai/home>.
- If you don't want wandb as your logger, set `CFG.LOG.WANDB` as `False` at `mdistiller/engine/cfg.py`.

### Training on CIFAR-100
```
python tools/train.py --cfg configs/cifar100/RLTKD/res56_res20.yaml
```
