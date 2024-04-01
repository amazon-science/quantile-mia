import os
import shutil
import warnings

warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import argparse
import random

# import time
# import math
import numpy as np
import pytorch_lightning as pl
import ray
import ray.tune as tune
import torch
from data_utils import CustomDataModule
from lightning_utils import LightningQMIA
from ray.air.config import CheckpointConfig
from ray.tune import CLIReporter

# from ray.train.torch import TorchTrainer
# from ray.train.lightning import (
#     RayDDPStrategy,
#     RayLightningEnvironment,
#     RayTrainReportCallback,
#     prepare_trainer
# )
from ray.tune.integration.pytorch_lightning import (
    TuneReportCheckpointCallback,
)  # TuneReportCallback,
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

# from pytorch_lightning.loggers.csv_logs import CSVLogger


NUM_CPUS_PER_GPU = 8
GPUS_PER_TRIAL = 1  # Current code does not support multi gpu per trial
NUM_CONCURRENT_TRIALS = 8


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
    parser.add_argument("--epochs", type=int, default=5, help="epochs")
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
    parser.add_argument(
        "--low_lr", type=float, default=1e-6, help="lower bound for learning rate"
    )
    parser.add_argument(
        "--high_lr", type=float, default=1e-3, help="lower bound for learning rate"
    )
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
        default="./models/",
        help="model directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/",
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


def train_func(config, num_gpus=1):
    # [1] Create a Lightning model
    model = LightningQMIA(
        architecture=args.architecture,
        base_architecture=args.base_architecture,
        image_size=args.image_size,
        hidden_dims=config["hidden_dims"],
        num_base_classes=args.num_base_classes,
        freeze_embedding=False,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        n_quantile=args.n_quantile,
        use_logscale=args.use_log_quantile,
        # cumulative_qr=config['cumulative_qr'],
        optimizer_params={"opt_type": args.opt, "scheduler": args.scheduler},
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
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        # weight_decay=0.,
        use_gaussian=args.use_gaussian,
        use_target_dependent_scoring=args.use_target_dependent_scoring,
        use_target_inputs=args.use_target_inputs,
    )

    # [2] Report Checkpoint with callback
    ckpt_report_callback = TuneReportCheckpointCallback(
        {
            "ptl/val_loss": "ptl/val_loss",
            "ptl/train_loss": "ptl/train_loss",
        },
        on="validation_end",
        filename="checkpoint",
    )

    # [3] Create a Lighting Trainer

    try:
        max_epochs = config["max_epochs"]
    except KeyError:
        max_epochs = args.epochs
    try:
        accumulate_grad_batches = config["accumulate_grad_batches"]
    except KeyError:
        accumulate_grad_batches = 1
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=args.grad_clip,
        # devices=math.ceil(num_gpus),
        callbacks=[ckpt_report_callback],
    )
    # trainer = trainer.as_trainable()

    # [4] Build your datasets on each worker
    datamodule = CustomDataModule(
        dataset_name=args.dataset,
        mode="mia",
        num_workers=NUM_CPUS_PER_GPU,
        image_size=args.image_size,
        batch_size=args.batch_size,
        data_root=args.data_root,
        use_augmentation=config["use_augmentation"],
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    args = argparser()

    metric = "ptl/val_loss"
    mode = "min"

    # Following
    # https://github.com/ray-project/ray/blob/master/doc/source/tune/examples/tune-vanilla-pytorch-lightning.ipynb
    # pip install "ray[tune]" torch torchvision pytorch_lightning
    scheduler = ASHAScheduler(max_t=args.epochs, grace_period=3, reduction_factor=2)
    search_alg = HyperOptSearch()
    resources_per_trial = {
        "cpu": int(NUM_CPUS_PER_GPU * GPUS_PER_TRIAL),
        "gpu": GPUS_PER_TRIAL,
    }

    config = {
        "lr": tune.loguniform(args.low_lr, args.high_lr),
        "hidden_dims": tune.choice([[], [512, 512]]),
        "weight_decay": tune.loguniform(args.weight_decay / 50, 50 * args.weight_decay),
        "use_augmentation": tune.choice([True, False]),
    }
    if args.epochs > 20 and args.epochs <= 100:
        config["max_epochs"] = tune.qrandint(5, args.epochs, 5)
    else:
        config["max_epochs"] = tune.qrandint(1, args.epochs, 1)
    if args.tune_batch_size:
        config["accumulate_grad_batches"] = tune.choice([1, 2, 4, 8, 16])

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["ptl/val_loss", "ptl/train_loss"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_func,
        num_gpus=GPUS_PER_TRIAL,
    )

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            scheduler=scheduler,
            num_samples=args.num_tune_samples,
            search_alg=search_alg,
            max_concurrent_trials=NUM_CONCURRENT_TRIALS,
            reuse_actors=False,
        ),
        run_config=ray.train.RunConfig(
            name="train_mia_ray",
            progress_reporter=reporter,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,
            ),
        ),
        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

    # Getting best trial and checkpoint combination
    best_results_array = []
    for r_idx, result in enumerate(results):
        metrics_dataframe = result.metrics_dataframe
        metrics = metrics_dataframe[metric].values
        if mode == "min":
            best_result_chkpoint = metrics.argmin()
        else:
            best_result_chkpoint = metrics.argmax()
        best_result_metric = metrics[best_result_chkpoint]
        best_results_array.append([best_result_metric, best_result_chkpoint, r_idx])

    best_results_array = np.array(best_results_array)
    best_result_idx = np.argmin(best_results_array[:, 0])
    best_result_abs_idx = best_results_array[best_result_idx, 2]
    best_result_chkpnt_idx = best_results_array[best_result_idx, 1]
    best_result = results[int(best_result_abs_idx)]

    # Moving best model
    src_checkpoint_path = os.path.join(
        os.path.dirname(best_result.checkpoint.path),
        "checkpoint_{:06d}".format(int(best_result_chkpnt_idx)),
        "checkpoint",
    )

    dst_checkpoint_path = os.path.join(args.root_checkpoint_path, "best_val_loss.ckpt")
    print(
        "Moved best result from {} to {}".format(
            src_checkpoint_path, dst_checkpoint_path
        )
    )
    os.makedirs(os.path.dirname(dst_checkpoint_path), exist_ok=True)
    shutil.copyfile(src_checkpoint_path, dst_checkpoint_path)

    # Removing
    for result in results:
        shutil.rmtree(result.path)

    print(" BEST TRIAL by VAL LOSS", best_result)
