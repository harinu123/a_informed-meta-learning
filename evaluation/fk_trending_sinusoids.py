from typing import Tuple

import torch

from models.loss import sum_log_prob


def parse_knowledge_trending(knowledge: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse knowledge tensor into parameter values and masks.

    Args:
        knowledge: Tensor shaped [B, R, 4] where R is the number of facts.

    Returns:
        k_val: Tensor shaped [B, 3] containing values for (a, b, c).
        k_mask: Tensor shaped [B, 3] with 1.0 where knowledge is present.
    """

    device = knowledge.device
    batch_size, num_rows, _ = knowledge.shape
    k_val = torch.zeros(batch_size, 3, device=device)
    k_mask = torch.zeros(batch_size, 3, device=device)

    for r in range(num_rows):
        one_hot = knowledge[:, r, :3]
        values = knowledge[:, r, 3]
        has_fact = torch.sum(one_hot, dim=1) > 0
        if not torch.any(has_fact):
            continue
        idx = torch.argmax(one_hot, dim=1)
        k_val[has_fact, idx[has_fact]] = values[has_fact]
        k_mask[has_fact, idx[has_fact]] = 1.0

    return k_val, k_mask


def fit_abc_from_curve(
    mu: torch.Tensor, x: torch.Tensor, b_grid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit (a, b, c) parameters for each particle's mean curve.

    Args:
        mu: Tensor of shape [M, B, T] containing mean predictions per particle.
        x: Tensor of shape [T] containing the target inputs.
        b_grid: Tensor of shape [J] with candidate frequency parameters.

    Returns:
        a_hat, b_hat, c_hat each of shape [M, B].
    """

    num_points = x.shape[0]
    mean_x = torch.mean(x)
    sum_x = torch.sum(x)
    sum_x2 = torch.sum(x**2)
    denom = sum_x2 - num_points * (mean_x**2)

    # [J, T]
    sin_bx = torch.sin(b_grid[:, None] * x[None, :])
    # y_prime: [M, B, J, T]
    y_prime = mu.unsqueeze(2) - sin_bx.unsqueeze(0).unsqueeze(0)

    sum_y = torch.sum(y_prime, dim=-1)  # [M, B, J]
    sum_xy = torch.sum(x.view(1, 1, 1, -1) * y_prime, dim=-1)  # [M, B, J]

    mean_y = sum_y / num_points
    a = (sum_xy - num_points * mean_x * mean_y) / denom
    c = mean_y - a * mean_x

    mu_hat = (
        a.unsqueeze(-1) * x.view(1, 1, 1, -1)
        + sin_bx.view(1, 1, *sin_bx.shape)
        + c.unsqueeze(-1)
    )
    sse = torch.sum((mu.unsqueeze(2) - mu_hat) ** 2, dim=-1)

    sse_min, best_idx = torch.min(sse, dim=2)
    # Avoid unused variable warning for sse_min
    _ = sse_min

    gather_idx = best_idx.unsqueeze(-1)
    a_hat = torch.gather(a, 2, gather_idx).squeeze(-1)
    b_hat = torch.gather(
        b_grid.view(1, 1, -1).expand_as(a), 2, gather_idx
    ).squeeze(-1)
    c_hat = torch.gather(c, 2, gather_idx).squeeze(-1)

    return a_hat, b_hat, c_hat


def compute_G(
    a_hat: torch.Tensor,
    b_hat: torch.Tensor,
    c_hat: torch.Tensor,
    k_val: torch.Tensor,
    k_mask: torch.Tensor,
    scales=(1.0, 6.0, 1.0),
) -> torch.Tensor:
    """Compute knowledge inconsistency score G for each particle.

    Args:
        a_hat, b_hat, c_hat: Tensors of shape [M, B].
        k_val, k_mask: Tensors of shape [B, 3].
        scales: Tuple of scaling factors for (a, b, c).

    Returns:
        Tensor of shape [M, B] with inconsistency scores.
    """

    params = torch.stack([a_hat, b_hat, c_hat], dim=-1)
    scales = torch.tensor(scales, device=params.device).view(1, 1, 3)
    diff = (params - k_val.unsqueeze(0)) / scales
    G = torch.sum(diff**2 * k_mask.unsqueeze(0), dim=-1)
    return G


def fk_steered_nll(
    outputs, y_target: torch.Tensor, x_target: torch.Tensor, knowledge: torch.Tensor, lam, b_grid: torch.Tensor
) -> torch.Tensor:
    """Compute FK-steered negative log likelihood for trending sinusoids.

    Args:
        outputs: Tuple (p_yCc, z_samples, q_zCc, q_zCct) from the model.
        y_target: Tensor [B, T, 1].
        x_target: Tensor [B, T, 1].
        knowledge: Tensor [B, R, 4] containing knowledge facts.
        lam: Steering strength (float >= 0).
        b_grid: Tensor [J] candidate b values.

    Returns:
        Tensor [B] with FK-steered NLL per batch element.
    """

    p_yCc, z_samples, q_zCc, q_zCct = outputs
    device = p_yCc.mean.device
    y_target = y_target.to(device)
    x_target = x_target.to(device)
    knowledge = knowledge.to(device)
    b_grid = b_grid.to(device)

    sum_log_p_yCz = sum_log_prob(p_yCc, y_target)  # [M, B]
    mu = p_yCc.mean.squeeze(-1)  # [M, B, T]

    x = x_target[0, :, 0]  # [T]
    k_val, k_mask = parse_knowledge_trending(knowledge)
    a_hat, b_hat, c_hat = fit_abc_from_curve(mu, x, b_grid)
    G = compute_G(a_hat, b_hat, c_hat, k_val, k_mask)

    log_w = -float(lam) * G
    log_num = torch.logsumexp(sum_log_p_yCz + log_w, dim=0)
    log_den = torch.logsumexp(log_w, dim=0)
    log_p = log_num - log_den

    nll = -log_p
    return nll
