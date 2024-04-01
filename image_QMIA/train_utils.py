import torch
from torchmetrics.utilities.data import to_onehot


##########
# distribution learning losses
##########
def pinball_loss_fn(score, target, quantile):
    target = target.reshape([-1, 1])
    assert (
        score.ndim == 2
    ), "score has the wrong shape, expected 2d input but got {}".format(score.shape)
    delta_score = target - score
    loss = torch.nn.functional.relu(delta_score) * quantile + torch.nn.functional.relu(
        -delta_score
    ) * (1.0 - quantile)
    return loss


def gaussian_loss_fn(score, target, quantile):
    # little different from the rest, score is Nx2, quantile is ignored, this is just a negative log likelihood of a Gaussian distribution
    assert (
        score.ndim == 2 and score.shape[-1] == 2
    ), "score has the wrong shape, expected Nx2 input but got {}".format(score.shape)
    assert (
        target.ndim == 1
    ), "target has the wrong shape, expected 1-d vector, got {}".format(target.shape)
    mu = score[:, 0]
    log_std = score[:, 1]
    assert (
        mu.shape == log_std.shape and mu.shape == target.shape
    ), "mean, std and target have non-compatible shapes, got {} {} {}".format(
        mu.shape, log_std.shape, target.shape
    )
    loss = log_std + 0.5 * torch.exp(-2 * log_std) * (target - mu) ** 2
    assert target.shape == loss.shape, "loss should be a 1-d vector got {}".format(
        loss.shape
    )
    return loss


##########
# Score functions for base network
##########


def label_logit_and_hinge_scoring_fn(samples, label, base_model):
    # z_y(x)-max_{y'\neq y} z_{y'}(x)
    base_model.eval()
    with torch.no_grad():
        logits = base_model(samples)

        oh_label = to_onehot(label, logits.shape[-1]).bool()
        score = logits[oh_label]
        score -= torch.max(logits[~oh_label].view(logits.shape[0], -1), dim=1)[0]
        assert (
            score.ndim == 1
        ), "hinge loss score should be 1-dimensional, got {}".format(score.shape)
    return score, logits


##########
# logit to quantile prediction nonlinearities for QR
##########


# Based on "Quantile and probability curves without crossing.", ensures that, at evaluation time, predicted quantiles are monotonically increasing (non differentiable)
# only usable for linearly spaced quantiles
def rearrange_quantile_fn(test_preds, all_quantiles, target_quantiles=None):
    """Produce monotonic quantiles
    Parameters
    ----------
    test_preds : array of predicted quantile (nXq)
    all_quantiles : array (q), grid of quantile levels in the range (0,1)
    target_quantiles: array (q'), grid of target quantile levels in the range (0,1)

    Returns
    -------
    q_fixed : array (nXq'), containing the rearranged estimates of the
              desired low and high quantile
    References
    ----------
    .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
            "Quantile and probability curves without crossing."
            Econometrica 78.3 (2010): 1093-1125.
    """
    if not target_quantiles:
        target_quantiles = all_quantiles

    scaling = all_quantiles[-1] - all_quantiles[0]
    rescaled_target_qs = (target_quantiles - all_quantiles[0]) / scaling
    q_fixed = torch.quantile(
        test_preds, rescaled_target_qs, interpolation="linear", dim=-1
    ).T
    assert (
        q_fixed.shape[0] == test_preds.shape[0] and q_fixed.ndim == test_preds.ndim
    ), "fixed quantiles have the wrong shape, {}".format(q_fixed.shape)
    return q_fixed
