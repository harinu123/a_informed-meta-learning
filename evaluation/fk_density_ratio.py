import torch

from models.loss import sum_log_prob


@torch.no_grad()
def _infer_q_zCc(model, x_context, y_context, x_target, knowledge):
    """Infer q(z|C,K) using the model's encoders.

    This mirrors the internal computations of ``INP.forward`` without sampling.
    """

    x_context_e = model.x_encoder(x_context)
    x_target_e = model.x_encoder(x_target)
    R = model.encode_globally(x_context_e, y_context, x_target_e)
    return model.infer_latent_dist(R, knowledge, x_context.shape[1])


@torch.no_grad()
def fk_density_ratio_nll(
    model,
    x_context,
    y_context,
    x_target,
    y_target,
    knowledge,
    alpha=1.0,
    num_samples=128,
):
    """FK steering using the density ratio log q(z|C,K) - log q(z|C,∅).

    Args:
        model: INP model.
        x_context, y_context, x_target, y_target: tensors for one batch.
        knowledge: knowledge tensor to use for the informed posterior.
        alpha: scaling on the density ratio.
        num_samples: number of z particles to draw from the null posterior.

    Returns:
        Tensor of shape [B] with FK-steered NLL values.
    """

    device = next(model.parameters()).device
    x_context = x_context.to(device)
    y_context = y_context.to(device)
    x_target = x_target.to(device)
    y_target = y_target.to(device)
    knowledge = knowledge.to(device)

    q_null = _infer_q_zCc(model, x_context, y_context, x_target, knowledge=None)
    q_informed = _infer_q_zCc(model, x_context, y_context, x_target, knowledge)

    z = q_null.rsample([num_samples])  # [M,B,1,D]
    log_w = alpha * (q_informed.log_prob(z) - q_null.log_prob(z))  # [M,B,1]
    log_w = log_w.squeeze(-1)

    p_yCc, _, _, _ = model(
        x_context,
        y_context,
        x_target,
        y_target=y_target,
        knowledge=None,
        z_samples=z,
    )
    sum_log_p = sum_log_prob(p_yCc, y_target)  # [M,B]

    log_num = torch.logsumexp(sum_log_p + log_w, dim=0)
    log_den = torch.logsumexp(log_w, dim=0)
    log_p = log_num - log_den

    return -log_p
