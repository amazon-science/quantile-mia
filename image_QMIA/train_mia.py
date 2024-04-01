import os
import warnings

warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import argparse
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch
from data_utils import CustomDataModule
from lightning_utils import LightningQMIA
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

NUM_CPUS_PER_WORKER = 7


def argparser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="QMIA attack")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # Options for pinball loss attack. Multi head regression where each head is a different quantile target
    parser.add_argument(
        "--n_quantile", type=int, default=41, help="number of quantile targets"
    )
    parser.add_argument(
        "--low_quantile",
        type=float,
        default=-4,
        help="minimum quantile, in absolute scale if use_log_quantile is false, otherwise just the exponent (0.01 vs -2)",
    )
    parser.add_argument(
        "--high_quantile",
        type=float,
        default=0,
        help="maximum quantile, in absolute scale if use_log_quantile is false, otherwise just the exponent (0.01 vs -2) ",
    )
    parser.add_argument(
        "--use_log_quantile",
        type=str2bool,
        default=True,
        help="use log scale for quantile sweep",
    )
    # Options for gaussian (mean, std) modelling of score distribution
    parser.add_argument(
        "--use_gaussian",
        type=str2bool,
        default=False,
        help="use gaussian parametrization",
    )
    # Optionally train a label-dependent regressor q(x,y) instead of q(x)
    parser.add_argument(
        "--use_target_dependent_scoring",
        type=str2bool,
        default=False,
        help="Use target label y for quantile predictor (q(x,y))?",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=30, help="epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="l2 regularization"
    )
    parser.add_argument(
        "--opt", type=str, default="adamw", help="otimizer {sgd, adam, adamw}"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="image input size, set to -1 to use dataset's default value",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="", help="learning rate scheduler"
    )
    # parser.add_argument('--grad_clip', type=float, default=0., help="gradient clipping")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="gradient clipping"
    )

    # QR model configuration
    parser.add_argument(
        "--architecture",
        type=str,
        default="facebook/convnext-tiny-224",
        help="Model Type",
    )
    parser.add_argument(
        "--model_name_prefix",
        type=str,
        default="bespoke",
        help="name prefix for output model (saving convention)",
    )

    # Base model configuration
    parser.add_argument(
        "--base_architecture",
        type=str,
        default="facebook/convnext-tiny-224",
        help="Model Type ",
    )
    parser.add_argument(
        "--base_model_name_prefix",
        type=str,
        default="bespoke",
        help="name prefix for base (classifier) model (saving convention)",
    )

    # Score configuration
    parser.add_argument(
        "--use_hinge_score",
        type=str2bool,
        default="True",
        help="use hinge loss of logits as score? otherwise use probability",
    )
    parser.add_argument(
        "--use_target_label",
        type=str2bool,
        default="True",
        help="use target label or argmax label of model output",
    )
    parser.add_argument(
        "--use_target_inputs",
        type=str2bool,
        default=False,
        help="use targets as input to the quantile model",
    )

    parser.add_argument(
        "--dataset", type=str, default="cifar100", help="dataset {'cifar10', 'mnist',}"
    )

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
    parser.add_argument(
        "--num_tune_samples", type=int, default=20, help="number of hyperparameter runs"
    )
    parser.add_argument(
        "--tune_batch_size",
        type=str2bool,
        default=False,
        help="tune batch size? (1x 2x 4x)",
    )
    parser.add_argument(
        "--return_mean_logstd",
        type=str2bool,
        default=False,
        help="just for plotting, stick to false",
    )

    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args.root_checkpoint_path = os.path.join(
        args.model_root,
        args.dataset,
        "mia",
        args.model_name_prefix,
        args.architecture,
        "use_hinge_{}".format(args.use_hinge_score),
        "use_target_{}".format(args.use_target_label),
    )

    return args


if __name__ == "__main__":
    args = argparser()

    start = time.time()
    if "cifar100" in args.dataset.lower():
        num_base_classes = 100
    elif "imagenet-1k" in args.dataset.lower():
        num_base_classes = 1000
    else:
        num_base_classes = 10

    metric = "ptl/val_loss"
    mode = "min"

    # Create lightning model
    lightning_model = LightningQMIA(
        architecture=args.architecture,
        base_architecture=args.base_architecture,
        image_size=args.image_size,
        hidden_dims=[512, 512],
        num_base_classes=num_base_classes,
        freeze_embedding=False,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        n_quantile=args.n_quantile,
        use_logscale=args.use_log_quantile,
        # cumulative_qr=False,
        optimizer_params={"opt_type": args.opt},
        base_model_path=os.path.join(
            args.model_root,
            args.dataset,
            "base",
            args.base_model_name_prefix,
            args.base_architecture,
            "model.pickle",
        ),
        rearrange_on_predict=not args.use_gaussian,
        use_hinge_score=args.use_hinge_score,
        use_target_label=args.use_target_label,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_gaussian=args.use_gaussian,
        use_target_dependent_scoring=args.use_target_dependent_scoring,
        use_target_inputs=args.use_target_inputs,
    )
    datamodule = CustomDataModule(
        dataset_name=args.dataset,
        mode="mia",
        num_workers=6,
        image_size=args.image_size,
        batch_size=args.batch_size,
        data_root=args.data_root,
    )
    metric = "ptl/val_loss"
    mode = "min"
    checkpoint_dir = os.path.dirname(args.root_checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.root_checkpoint_path,
        monitor=metric,
        mode=mode,
        save_top_k=1,
        auto_insert_metric_name=False,
        filename="best_val_loss",
    )
    callbacks = [checkpoint_callback] + [TQDMProgressBar(refresh_rate=100)]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        callbacks=callbacks,
        devices=-1,
        gradient_clip_val=args.grad_clip,
        default_root_dir=os.path.join(args.root_checkpoint_path, "tune"),
    )
    trainer.fit(lightning_model, datamodule=datamodule)

    print(checkpoint_callback.best_model_path)
