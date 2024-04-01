import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

# Get base quantile performances


def get_rates(
    private_target_scores, public_target_scores, private_thresholds, public_thresholds
):
    # Get TPR, TNR and precision for all thresholds
    # scores are real valued vectors of size n
    # thresholds are either [n,n_thresholds] or [1,n_thresholds] depending on if the threshold is sample dependent or not
    assert (
        len(private_target_scores.shape) == 1
    ), "private scores need to be real-valued vectors"
    assert (
        len(public_target_scores.shape) == 1
    ), "public scores need to be real-valued vectors"
    assert (
        len(private_thresholds.shape) == 2
    ), "private thresholds need to be 2-d vectors"
    assert len(public_thresholds.shape) == 2, "public thresholds need to be 2-d vectors"
    prior = 0.0

    true_positives = (private_target_scores.reshape([-1, 1]) >= private_thresholds).sum(
        0
    ) + prior
    false_negatives = (private_target_scores.reshape([-1, 1]) < private_thresholds).sum(
        0
    ) + prior
    true_negatives = (public_target_scores.reshape([-1, 1]) < public_thresholds).sum(
        0
    ) + prior
    false_positives = (public_target_scores.reshape([-1, 1]) >= public_thresholds).sum(
        0
    ) + prior

    true_positive_rate = np.nan_to_num(
        true_positives / (true_positives + false_negatives)
    )
    true_negative_rate = np.nan_to_num(
        true_negatives / (true_negatives + false_positives)
    )
    precision = np.nan_to_num(
        true_positive_rate / (true_positive_rate + 1 - true_negative_rate)
    )

    return precision, true_positive_rate, true_negative_rate


def pinball_loss_np(target, score, quantile):
    target = target.reshape([-1, 1])
    assert (
        score.ndim == 2
    ), "score has the wrong shape, expected 2d input but got {}".format(score.shape)
    delta_score = target - score
    loss = np.maximum(delta_score * quantile, -delta_score * (1.0 - quantile)).mean(0)
    return loss


def plot_performance_curves(
    private_target_scores,
    public_target_scores,
    private_predicted_score_thresholds=None,
    public_predicted_score_thresholds=None,
    model_target_quantiles=None,
    model_name="Quantile Model",
    use_quantile_thresholds=True,
    use_thresholds=True,
    use_logscale=True,
    fontsize=12,
    savefig_path="results.png",
    plot_results=True,
):
    plt.ioff()
    n_baseline_points = 500
    if use_quantile_thresholds:
        if use_logscale:
            baseline_quantiles = np.sort(
                1.0 - np.logspace(-6, 0, n_baseline_points)[:-1]
            )
        else:
            baseline_quantiles = np.linspace(0, 1, n_baseline_points)[:-1]
        baseline_thresholds = np.quantile(public_target_scores, baseline_quantiles)
        baseline_public_loss = pinball_loss_np(
            public_target_scores,
            baseline_thresholds.reshape([1, -1]),
            baseline_quantiles,
        )
        baseline_private_loss = pinball_loss_np(
            private_target_scores,
            baseline_thresholds.reshape([1, -1]),
            baseline_quantiles,
        )

    else:
        raise NotImplementedError

    baseline_precision, baseline_tpr, baseline_tnr = get_rates(
        private_target_scores,
        public_target_scores,
        baseline_thresholds.reshape([1, -1]),
        baseline_thresholds.reshape([1, -1]),
    )

    (
        model_precision,
        model_tpr,
        model_tnr,
        model_auc,
        model_public_loss,
        model_private_loss,
    ) = (None, None, None, None, None, None)

    if (
        private_predicted_score_thresholds is not None and use_thresholds
    ):  # scores and thresholds are provided directly (quantile model)
        model_target_quantiles = np.sort(model_target_quantiles)
        private_predicted_score_thresholds = np.sort(
            private_predicted_score_thresholds, axis=-1
        )
        public_predicted_score_thresholds = np.sort(
            public_predicted_score_thresholds, axis=-1
        )

        model_precision, model_tpr, model_tnr = get_rates(
            private_target_scores,
            public_target_scores,
            private_predicted_score_thresholds,
            public_predicted_score_thresholds,
        )
        model_public_loss = pinball_loss_np(
            public_target_scores,
            public_predicted_score_thresholds,
            model_target_quantiles,
        )
        model_private_loss = pinball_loss_np(
            private_target_scores,
            private_predicted_score_thresholds,
            model_target_quantiles,
        )

        model_adjusted_public_loss = pinball_loss_np(
            public_target_scores, public_predicted_score_thresholds, model_tnr
        )

    # Plot ROC
    fig, ax = plt.subplots(figsize=(6, 6), ncols=1, nrows=1)

    ax.set_title("ROC", fontsize=fontsize)
    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")
    ax.set_ylim([1e-3, 1])
    ax.set_xlim([1e-3, 1])
    baseline_auc = np.abs(np.trapz(baseline_tpr, x=1 - baseline_tnr))
    # baseline_acc = (baseline_tpr + baseline_tnr).max() / 2.0
    ax.plot(
        1 - baseline_tnr,
        baseline_tpr,
        "-",
        # label="Marginal Quantiles Acc {:.1f}%".format(100 * baseline_max_acc),
        label="Marginal Quantiles",
    )
    if model_tpr is not None:
        model_auc = np.abs(np.trapz(model_tpr, x=1 - model_tnr))
        # model_acc = (model_tpr + model_tnr).max() / 2.0
        ax.plot(
            1 - model_tnr,
            model_tpr,
            "-",
            markersize=12,
            # label="{} Acc {:.1f}%".format(model_name, 100 * model_acc),
            label="{}".format(model_name),
        )

    ax.legend()
    if use_logscale:
        plt.semilogx()
        plt.semilogy()
    # Finishing
    plt.tight_layout()
    if savefig_path is not None:
        os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
        roc_path = os.path.join(os.path.dirname(savefig_path), "roc.png")
        plt.savefig(roc_path, dpi=300)
        print("saving plot to", roc_path)
    if plot_results:
        plt.show()

    # Plot Pinball losses on public data
    fig, ax = plt.subplots(figsize=(6, 6), ncols=1, nrows=1)

    ax.set_title("Pinball loss", fontsize=fontsize)
    ax.set_xlabel("Significance level")
    ax.set_ylabel("Pinball loss")
    color = next(ax._get_lines.prop_cycler)["color"]
    ax.plot(
        1 - baseline_quantiles,
        baseline_public_loss,
        "x-",
        label="Marginal Quantiles" + " (Public)",
        color=color,
    )
    if model_public_loss is not None:
        color = next(ax._get_lines.prop_cycler)["color"]
        ax.plot(
            1 - model_target_quantiles,
            model_public_loss,
            "x-",
            label=model_name + "  (Public)",
            color=color,
        )
    plt.semilogx()
    ax.legend()
    # Finishing
    plt.tight_layout()
    if savefig_path is not None:
        os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
        pinball_path = os.path.join(os.path.dirname(savefig_path), "pinball.png")
        plt.savefig(pinball_path, dpi=300)
        print("saving plot to", pinball_path)
    if plot_results:
        plt.show()

    # pickle results and also print results at 1% and 0.1% FPR
    pickle_path = os.path.join(
        os.path.dirname(savefig_path),
        os.path.basename(savefig_path).split(".")[0] + ".pkl",
    )

    def convenience_dict(
        model_precision,
        model_tpr,
        model_tnr,
        model_auc,
        model_public_loss,
        model_private_loss,
        adjusted_public_loss=None,
    ):
        idx_1pc = np.argmin(np.abs(model_tnr - 0.99))
        idx_01pc = np.argmin(np.abs(model_tnr - 0.999))
        print(
            "Precision @1%  FPR {:.2f}% \t  TPR @ 1% FPR {:.2f}% ".format(
                model_precision[idx_1pc] * 100, model_tpr[idx_1pc] * 100
            )
        )
        print(
            "Precision @0.1% FPR {:.2f}% \t  TPR @ 0.1% FPR {:.2f}% ".format(
                model_precision[idx_01pc] * 100, model_tpr[idx_01pc] * 100
            )
        )
        cdict = {
            "precision": model_precision,
            "tpr": model_tpr,
            "tnr": model_tnr,
            "auc": model_auc,
            "public_loss": model_public_loss,
            "private_loss": model_private_loss,
        }
        cdict["adjusted_public_loss"] = (
            adjusted_public_loss
            if adjusted_public_loss is not None
            else model_public_loss
        )
        return cdict

    with open(pickle_path, "wb") as f:
        save_dict = {}
        if baseline_tnr is not None:
            print("baseline")
            save_dict["baseline"] = convenience_dict(
                baseline_precision,
                baseline_tpr,
                baseline_tnr,
                baseline_auc,
                baseline_public_loss,
                baseline_private_loss,
            )

        if model_tpr is not None:
            print("model")
            save_dict["model"] = convenience_dict(
                model_precision,
                model_tpr,
                model_tnr,
                model_auc,
                model_public_loss,
                model_private_loss,
                model_adjusted_public_loss,
            )
        pickle.dump(save_dict, f)

    return baseline_auc, model_auc
