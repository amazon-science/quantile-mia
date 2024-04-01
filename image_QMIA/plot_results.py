import os
import shutil
import warnings

warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import math
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
from analysis_utils import plot_performance_curves
from data_utils import CustomDataModule
from lightning_utils import CustomWriter, LightningQMIA
from train_mia_ray import argparser


def plot_model(
    args,
    checkpoint_path,
    fig_name="best",
    recompute_predictions=True,
    return_mean_logstd=False,
):
    if return_mean_logstd:
        fig_name = "raw_{}".format(fig_name)
        prediction_output_dir = os.path.join(
            args.root_checkpoint_path,
            "raw_predictions",
            fig_name,
        )
    else:
        prediction_output_dir = os.path.join(
            args.root_checkpoint_path,
            "predictions",
            fig_name,
        )
    print("Saving predictions to", prediction_output_dir)

    os.makedirs(prediction_output_dir, exist_ok=True)

    if (
        recompute_predictions
        or len(glob(os.path.join(prediction_output_dir, "*.pt"))) == 0
    ):
        try:
            if os.environ["LOCAL_RANK"] == "0":
                shutil.rmtree(prediction_output_dir)
        except:
            pass
        # os.makedirs(prediction_output_dir, exist_ok=True)
        # Get model and data
        datamodule = CustomDataModule(
            dataset_name=args.dataset,
            mode="eval",
            num_workers=7,
            image_size=args.image_size,
            batch_size=args.batch_size,
            data_root=args.data_root,
        )

        # reload quantile model
        print("reloading from", checkpoint_path)
        lightning_model = LightningQMIA.load_from_checkpoint(checkpoint_path)
        if return_mean_logstd:
            lightning_model.return_mean_logstd = True
        pred_writer = CustomWriter(
            output_dir=prediction_output_dir, write_interval="epoch"
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto" if torch.cuda.is_available() else "cpu",
            callbacks=[pred_writer],
            devices=1,
            enable_progress_bar=True,
        )
        predict_results = trainer.predict(
            lightning_model, datamodule, return_predictions=True
        )
        trainer.strategy.barrier()
        if trainer.global_rank != 0:
            return

    # Trainer predict in DDP does not return predictions. To use distributed predicting, we instead save the prediciton outputs to file then concatenate manually
    predict_results = None
    for file in glob(os.path.join(prediction_output_dir, "*.pt")):
        rank_predict_results = torch.load(file)
        if predict_results is None:
            predict_results = rank_predict_results
        else:
            for r, p in zip(rank_predict_results, predict_results):
                p.extend(r)

    def join_list_of_tuples(list_of_tuples):
        n_tuples = len(list_of_tuples[0])
        result = []
        for _ in range(n_tuples):
            try:
                result.append(torch.concat([t[_] for t in list_of_tuples]))
            except:
                result.append(torch.Tensor([t[_] for t in list_of_tuples]))
        return result

    (
        private_predicted_quantile_threshold,
        private_target_score,
        private_loss,
        private_base_acc1,
        private_base_acc5,
    ) = join_list_of_tuples(predict_results[-1])
    (
        test_predicted_quantile_threshold,
        test_target_score,
        test_loss,
        test_base_acc1,
        test_base_acc5,
    ) = join_list_of_tuples(predict_results[1])

    model_target_quantiles = np.sort(
        1.0
        - np.logspace(args.low_quantile, args.high_quantile, args.n_quantile).flatten()
        if args.use_log_quantile
        else np.linspace(
            args.low_quantile, args.high_quantile, args.n_quantile
        ).flatten()
    )
    if return_mean_logstd:
        # model_target_quantiles = model_target_quantiles[1:-1]
        dislocated_quantiles = torch.erfinv(
            2 * torch.Tensor(model_target_quantiles) - 1
        ).reshape([1, -1]) * math.sqrt(2)

        public_mu = test_predicted_quantile_threshold[:, 0].reshape([-1, 1])
        public_std = torch.exp(test_predicted_quantile_threshold[:, 1]).reshape([-1, 1])
        test_predicted_quantile_threshold = (
            public_mu + public_std * dislocated_quantiles
        )

        private_mu = private_predicted_quantile_threshold[:, 0].reshape([-1, 1])
        private_std = torch.exp(private_predicted_quantile_threshold[:, 1]).reshape(
            [-1, 1]
        )
        private_predicted_quantile_threshold = (
            private_mu + private_std * dislocated_quantiles
        )
    print(
        "Model accuracy on training set {:.2f}%".format(
            np.mean(private_base_acc1.numpy())
        )
    )
    print("Model accuracy on test set  {:.2f}%".format(np.mean(test_base_acc1.numpy())))

    plot_result = plot_performance_curves(
        np.asarray(private_target_score),
        np.asarray(test_target_score),
        private_predicted_score_thresholds=np.asarray(
            private_predicted_quantile_threshold
        ),
        public_predicted_score_thresholds=np.asarray(test_predicted_quantile_threshold),
        model_target_quantiles=model_target_quantiles,
        model_name="Quantile Regression",
        use_logscale=True,
        fontsize=12,
        savefig_path="./plots/{}/{}/{}/ray/use_hinge_{}/use_target_{}/{}.png".format(
            args.model_name_prefix + args.dataset,
            args.base_architecture.replace("/", "_"),
            args.architecture.replace("/", "_"),
            args.use_hinge_score,
            args.use_target_label,
            fig_name,
        ),
    )
    return plot_result


if __name__ == "__main__":
    args = argparser()
    dst_checkpoint_path = os.path.join(args.root_checkpoint_path, "best_val_loss.ckpt")

    # plot best trial
    plot_model(
        args,
        dst_checkpoint_path,
        "best",
        recompute_predictions=False,
        return_mean_logstd=args.return_mean_logstd,
    )
