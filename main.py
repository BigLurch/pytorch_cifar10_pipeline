import argparse
from pathlib import Path

from torchvision.datasets import CIFAR10

from src.k2.dataset import DataConfig, make_dataloaders
from src.k2.eval import eval_checkpoint
from src.k2.model import SimpleCNN
from src.k2.train import TrainConfig, train


def download_cifar10(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    CIFAR10(root=str(data_dir), train=True, download=True)
    CIFAR10(root=str(data_dir), train=False, download=True)
    print(f"✅ CIFAR-10 downloaded to: {data_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="K2 PyTorch pipeline (CIFAR-10)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download-data", help="Download CIFAR-10 dataset")
    p_dl.add_argument("--data-dir", type=Path, default=Path("dl/cifar10"))

    p_train = sub.add_parser("train", help="Train baseline model")
    p_train.add_argument("--data-dir", type=Path, default=Path("dl/cifar10"))
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=0.0)
    p_train.add_argument("--run-name", type=str, default="exp1")
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("eval", help="Evaluate a saved checkpoint")
    p_eval.add_argument("--data-dir", type=Path, default=Path("dl/cifar10"))
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--ckpt", type=Path, required=True)
    p_eval.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.cmd == "download-data":
        download_cifar10(args.data_dir)
        return

    if args.cmd == "train":
        dcfg = DataConfig(data_dir=args.data_dir, batch_size=args.batch_size)
        train_loader, test_loader = make_dataloaders(dcfg)

        model = SimpleCNN(num_classes=10)
        tcfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            run_name=args.run_name,
            device=args.device,
            seed=args.seed,
        )
        out = train(model, train_loader, test_loader, tcfg)
        print(f"✅ Done. best_acc={out['best_acc']:.4f} ckpt={out['checkpoint']}")
        return

    if args.cmd == "eval":
        dcfg = DataConfig(data_dir=args.data_dir, batch_size=args.batch_size)
        _, test_loader = make_dataloaders(dcfg)

        model = SimpleCNN(num_classes=10)
        metrics = eval_checkpoint(model, test_loader, args.ckpt, device=args.device)
        print(f"✅ Eval: loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}")
        return

    raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
