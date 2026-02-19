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
git clone https://github.com/BigLurch/pytorch_cifar10_pipeline.git
cd pytorch_cifar10_pipeline
```

### 2. Klona repot

```bash
uv sync
```

### 3. Hämta dataset (DVC)

```bash
uv run dvc pull
```

## Köra pipeline

### 1. Ladda ner data (om du inte kör DVC pull)

```bash
uv run python main.py download-data --data-dir dl/cifar10
```

### 2. Träna baseline

```bash
uv run python main.py train --run-name exp1 --epochs 3 --lr 1e-3 --batch-size 128
```

### 3. Evaluera checkpoint

```bash
uv run python main.py eval --ckpt outputs/exp1/best.pt
```
