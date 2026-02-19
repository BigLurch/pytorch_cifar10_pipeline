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

## Experiment

| Experiment | Config    | Learning Rate | Batch Size | Weight Decay | Epochs | Best Accuracy |
| ---------- | --------- | ------------- | ---------- | ------------ | ------ | ------------- |
| Exp 1      | exp1.yaml | 0.0010        | 128        | 0.0          | 3      | 0.6788        |
| Exp 2      | exp2.yaml | 0.0030        | 128        | 0.0005       | 3      | 0.6310        |
| Exp 3      | exp3.yaml | 0.0007        | 256        | 0.0005       | 3      | 0.5966        |

### Köra experiment via config

```bash
uv run python main.py train --config configs/exp1.yaml
uv run python main.py train --config configs/exp2.yaml
uv run python main.py train --config configs/exp3.yaml
```
