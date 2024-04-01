import os
import warnings

warnings.simplefilter("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import argparse
import random

import numpy as np
import pytorch_lightning as pl
import torch
from data_utils import CustomDataModule

# from lightning_utils import BestMetricsLoggerCallback, LightningBaseNet
from lightning_utils import LightningBaseNet
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

NUM_CPUS_PER_WORKER = 7


def argparser():
    parser = argparse.ArgumentParser(description="Base network trainer")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--min_factor",
        type=float,
        default=0.3,
        help="minimum learning rate factor for linear/cosine scheduler",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="l2 regularization"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="epochs"
    )  # For small batch
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="image input size, set to -1 to use dataset's default value",
    )

    parser.add_argument(
        "--architecture", type=str, default="cifar-resnet-50", help="Model Type "
    )
    parser.add_argument("--model_name_prefix", type=str, default="smallbatch", help="")
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer")
    parser.add_argument(
        "--scheduler", type=str, default="step", help="learning rate scheduler"
    )
    parser.add_argument(
        "--scheduler_step_gamma",
        type=float,
        default=0.2,
        help="scheduler reduction fraction for step scheduler",
    )
    parser.add_argument(
        "--scheduler_step_fraction",
        type=float,
        default=0.3,
        help="scheduler fraction of steps between decays",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=0.0, help="gradient clipping"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="label_smoothing"
    )
    parser.add_argument("--dataset", type=str, default="cinic10/0_16", help="dataset")
    parser.add_argument(
        "--model_root",
        type=str,
        default="/dataXL/membership_inference/workspace/torch/lightning_models",
        help="model directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/dataXL/membership_inference/workspace/data/",
        help="dataset root directory",
    )
    args = parser.parse_args()

    args.base_checkpoint_path = os.path.join(
        args.model_root, args.dataset, "base", args.model_name_prefix, args.architecture
    )

    if "cifar100" in args.dataset.lower():
        args.num_base_classes = 100
    elif "imagenet-1k" in args.dataset.lower():
        args.num_base_classes = 1000
    else:
        args.num_base_classes = 10

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return args


def train_model(config, args, callbacks=None, rerun=False):
    callbacks = callbacks or []
    save_handle = "model.pickle"
    checkpoint_path = os.path.join(args.base_checkpoint_path, save_handle)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if (
        os.path.exists(checkpoint_path) and not rerun
    ):  # simple safeguard to only train each split once
        print("skipping")
        return

    # # This pretrains a classification model on a dataset to use as a model to run a QMIA attack on

    # Get Dataset
    datamodule = CustomDataModule(
        dataset_name=args.dataset,
        mode="base",
        num_workers=6,
        image_size=args.image_size,
        batch_size=args.batch_size,
        data_root=args.data_root,
    )

    # Create lightning model
    lightning_model = LightningBaseNet(
        architecture=args.architecture,
        num_classes=args.num_base_classes,
        optimizer_params=config,
        label_smoothing=config["label_smoothing"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="ptl/val_acc1",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
        filename="best",
    )
    callbacks = callbacks + [checkpoint_callback] + [TQDMProgressBar(refresh_rate=1000)]

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu",
        callbacks=callbacks,
        devices=-1,
        default_root_dir=checkpoint_dir,
        gradient_clip_val=config["gradient_clip_val"],
    )

    # Pretrain base network
    trainer.logger.log_hyperparams(config)
    if trainer.global_rank == 0:
        print(args.dataset)
    trainer.fit(lightning_model, datamodule=datamodule)
    if trainer.global_rank == 0:
        # reload best network and save just the base model
        lightning_model = LightningBaseNet.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        torch.save(lightning_model.model.state_dict(), checkpoint_path)
        print(
            "saved model from {} to {} ".format(
                checkpoint_callback.best_model_path, checkpoint_path
            )
        )
    trainer.strategy.barrier()


if __name__ == "__main__":
    args = argparser()

    config = {
        "lr": args.lr,
        "scheduler": args.scheduler,
        "min_factor": args.min_factor,
        "epochs": args.epochs,
        "opt_type": args.optimizer,
        "weight_decay": args.weight_decay,
        "step_gamma": args.scheduler_step_gamma,
        "step_fraction": args.scheduler_step_fraction,
        "gradient_clip_val": args.grad_clip,
        "label_smoothing": args.label_smoothing,
    }

    train_model(config, args, callbacks=None, rerun=True)
