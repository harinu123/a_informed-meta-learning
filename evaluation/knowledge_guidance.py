import torch

from models.utils import MultivariateNormalDiag

_EPS = 1e-6


def collapse_mc_diag_gaussian(p_y):
    """
    p_y: Independent(Normal) with batch shape [n_z, bs, n_pts, out_dim]
    Returns:
      mu  [bs, n_pts, out_dim]
      var [bs, n_pts, out_dim]
    """
    mu_z = p_y.mean
    var_z = p_y.stddev**2

    mu = mu_z.mean(dim=0)
    m2 = (var_z + mu_z**2).mean(dim=0)
    var = (m2 - mu**2).clamp_min(_EPS)
    return mu, var


def guided_diag_gaussian(mu0, var0, mu1, var1, s):
    """
    mu0,var0,mu1,var1: [bs, n_pts, out_dim]
    s: [bs, 1, 1] or [bs] (will be broadcast)
    Returns mu_s,var_s same shape as mu0
    """
    if s.dim() == 1:
        s = s[:, None, None]
    s = s.clamp(0.0, 1.0)

    tau0 = 1.0 / var0
    tau1 = 1.0 / var1
    taus = (1.0 - s) * tau0 + s * tau1
    vars_ = (1.0 / taus).clamp_min(_EPS)
    mus = ((1.0 - s) * tau0 * mu0 + s * tau1 * mu1) / taus
    return mus, vars_


def make_repeated_dist(mu, var, num_z):
    """
    mu,var: [bs, n_pts, out_dim]
    Return dist with batch shape [num_z, bs, n_pts, out_dim]
    """
    mu_z = mu.unsqueeze(0).expand(num_z, -1, -1, -1)
    std_z = var.sqrt().unsqueeze(0).expand(num_z, -1, -1, -1)
    return MultivariateNormalDiag(mu_z, std_z)


def optimize_s_from_cal(
    mu0, var0, mu1, var1, y_cal, steps=15, lr=0.2, s0=0.2, prior_w=0.01
):
    """
    mu0,var0,mu1,var1: [bs, n_cal, out_dim] for p0/p1 on calibration inputs
    y_cal:             [bs, n_cal, out_dim]
    Returns s*: [bs]
    """
    with torch.enable_grad():
        bs = y_cal.shape[0]
        logit_s = torch.zeros(bs, device=y_cal.device, requires_grad=True)
        opt = torch.optim.Adam([logit_s], lr=lr)

        for _ in range(steps):
            s = torch.sigmoid(logit_s)
            mu_s, var_s = guided_diag_gaussian(mu0, var0, mu1, var1, s)

            nll = 0.5 * (
                ((y_cal - mu_s) ** 2) / var_s + torch.log(var_s) + 1.8378770664093453
            )
            nll = nll.sum(dim=(1, 2))

            reg = prior_w * (s - s0).pow(2)
            loss = (nll + reg).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        return torch.sigmoid(logit_s).detach()


def guided_forward(
    model,
    x_cond,
    y_cond,
    x_cal,
    y_cal,
    x_target,
    y_target,
    knowledge,
    steps=15,
    lr=0.2,
    s0=0.2,
    prior_w=0.01,
):
    """
    Returns:
      outputs_guided tuple: (p_guided, z_samples_dummy, q_zCc_dummy, q_zCct_dummy)
      s_star: [bs]
    Notes:
      - does not train/grad model params
      - relies only on model forward passes for p0/p1
    """
    model.eval()

    with torch.no_grad():
        out0_cal = model(x_cond, y_cond, x_cal, y_target=y_cal, knowledge=None)
        out1_cal = model(x_cond, y_cond, x_cal, y_target=y_cal, knowledge=knowledge)

        mu0_cal, var0_cal = collapse_mc_diag_gaussian(out0_cal[0])
        mu1_cal, var1_cal = collapse_mc_diag_gaussian(out1_cal[0])

    s_star = optimize_s_from_cal(
        mu0_cal,
        var0_cal,
        mu1_cal,
        var1_cal,
        y_cal,
        steps=steps,
        lr=lr,
        s0=s0,
        prior_w=prior_w,
    )

    with torch.no_grad():
        out0_t = model(x_cond, y_cond, x_target, y_target=y_target, knowledge=None)
        out1_t = model(x_cond, y_cond, x_target, y_target=y_target, knowledge=knowledge)

        mu0_t, var0_t = collapse_mc_diag_gaussian(out0_t[0])
        mu1_t, var1_t = collapse_mc_diag_gaussian(out1_t[0])

        mu_s, var_s = guided_diag_gaussian(mu0_t, var0_t, mu1_t, var1_t, s_star)

        num_z = out1_t[0].mean.shape[0]
        p_guided = make_repeated_dist(mu_s, var_s, num_z=num_z)

        outputs_guided = (p_guided, out1_t[1], out1_t[2], out1_t[3])

    return outputs_guided, s_star
