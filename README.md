# PyTorch CIFAR-10 Pipeline

Modulär och reproducerbar Deep Learning-pipeline byggd i PyTorch med:

- CIFAR-10 dataset
- DVC för dataversionering
- uv för dependency management
- CLI-baserad main.py
- Reproducerbar träning

## Setup

### 1. Klona repot

```bash
git clone <repo-url>
cd pytorch_cifar10_pipeline
```

### 2. Klona repot

```bash
uv sync
```

## Hämta dataset (DVC)

```bash
uv run dvc pull
```
