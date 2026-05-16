# toy_cld.py
# Minimal pedagogical Critically-Damped Langevin Diffusion style model.
#
# Data: 2D Gaussian mixture
# State: z = (x, v), where x in R^2, v in R^2
# Forward CLD-like SDE:
#
#   dx = beta v dt
#   dv = -beta x dt - gamma beta v dt + sigma dW
#
# with gamma = 2, beta = 4, sigma = sqrt(2 gamma beta).
#
# The neural net learns the velocity score:
#
#   s_theta(t, x_t, v_t) ≈ ∇_{v_t} log p_t(v_t | x_t)
#
# For simplicity, we train with the paper's hybrid score matching idea:
# sample x0 from data, analytically marginalize over v0 ~ N(0, I), and learn
# the residual scaled velocity noise corresponding to p_t(x_t, v_t | x0). At
# sampling time this is combined with an analytic Gaussian score term and
# converted back into a velocity score.

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# -----------------------------
# Config
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

D = 2  # data dimension
ZDIM = 2 * D  # phase-space dimension: (x, v)
gamma = 2.0  # critical damping for unit oscillator
beta = 4.0  # paper-style time rescaling; beta * T matches the old T=4 horizon
sigma = math.sqrt(2.0 * gamma * beta)
T = 1.0
eps_t = 1e-2

batch_size = 512
num_steps = 100_000
lr = 2e-4
data_radius = 2.0
data_std = 0.10


# -----------------------------
# Dummy data distribution
# -----------------------------


def data_centers() -> torch.Tensor:
    angles = torch.arange(8, device=device) * (2.0 * math.pi / 8.0)
    return data_radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def sample_data(n: int) -> torch.Tensor:
    """
    Eight Gaussian modes equally spaced on a 2D ring.
    Returns x0 with shape [n, 2].
    """
    centers = data_centers()
    ids = torch.randint(0, len(centers), (n,), device=device)
    x = centers[ids] + data_std * torch.randn(n, D, device=device)
    return x


# -----------------------------
# Linear CLD transition
# -----------------------------


def make_matrices():
    """
    Build F and G for the linear SDE

        dz = F z dt + G dW

    with z = [x, v].
    """
    I = torch.eye(D, device=device)
    Z = torch.zeros(D, D, device=device)

    Fmat = torch.cat(
        [
            torch.cat([Z, beta * I], dim=1),
            torch.cat([-beta * I, -gamma * beta * I], dim=1),
        ],
        dim=0,
    )

    Gmat = torch.cat(
        [
            torch.zeros(D, D, device=device),
            sigma * I,
        ],
        dim=0,
    )

    Q = Gmat @ Gmat.T
    return Fmat, Gmat, Q


Fmat, Gmat, Qmat = make_matrices()
P_EQ = torch.eye(ZDIM, device=device)


def make_reverse_a_matrix():
    """
    Linear part A of the reverse CLD sampler, excluding the score correction.

    The full reverse drift is
        dx = -beta v dt
        dv = beta (x + gamma v + 2 gamma score_v) dt.

    SSCS splits this into an analytically solvable Langevin part
        dx = -beta v dt
        dv = beta (x - gamma v) dt + sigma dW
    and a score/friction correction
        dv = sigma^2 (score_v + v) dt
    for the critically damped M=1, gamma=2 setup used here.
    """
    I = torch.eye(D, device=device)
    Z = torch.zeros(D, D, device=device)
    return torch.cat(
        [
            torch.cat([Z, -beta * I], dim=1),
            torch.cat([beta * I, -gamma * beta * I], dim=1),
        ],
        dim=0,
    )


Fbar_amat = make_reverse_a_matrix()


def process_covariance(A: torch.Tensor):
    """
    Exact process-noise covariance for this linear CLD.

    Since the equilibrium covariance is I for x and v in this toy setting,
    Sigma_t = P_EQ - A P_EQ A^T.
    """
    if A.ndim == 2:
        cov = P_EQ - A @ P_EQ @ A.T
    else:
        p_eq = P_EQ.expand(A.shape[0], -1, -1)
        cov = p_eq - A @ p_eq @ A.transpose(-1, -2)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    return cov + 1e-6 * P_EQ.expand_as(cov)


def transition_mean_cov(z0: torch.Tensor, t: torch.Tensor):
    """
    Compute exact conditional transition

        z_t | z_0 ~ N(exp(F t) z_0, Sigma_t)

    For this toy version, t is a scalar tensor shared by the whole batch.
    """
    assert t.ndim == 0, "For this simple demo, use one scalar t per batch."

    A = torch.matrix_exp(Fmat * t)
    mean = z0 @ A.T
    cov = process_covariance(A)
    return mean, cov


def _initial_covariance(x0_var: float = 0.0, v0_var: float = 1.0) -> torch.Tensor:
    init_cov = torch.zeros(ZDIM, ZDIM, device=device)
    init_cov[:D, :D] = x0_var * torch.eye(D, device=device)
    init_cov[D:, D:] = v0_var * torch.eye(D, device=device)
    return init_cov


def hsm_mean_cov(x0: torch.Tensor, t: torch.Tensor, v0_var: float = 1.0):
    """
    HSM perturbation kernel p_t(x_t, v_t | x0), marginalizing v0.

    This is the key paper objective difference from plain DSM: v0 is not sampled
    and conditioned on as a sharp initial point. Its Gaussian distribution is
    integrated into the transition covariance.
    """
    z0_mean = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
    init_cov = _initial_covariance(v0_var=v0_var)

    if t.ndim == 0:
        A = torch.matrix_exp(Fmat * t)
        mean = z0_mean @ A.T
        cov = process_covariance(A) + A @ init_cov @ A.T
    else:
        t_flat = t.reshape(-1)
        if t_flat.shape[0] != x0.shape[0]:
            raise ValueError("Batched t must have one value per x0 sample.")
        A = torch.matrix_exp(t_flat[:, None, None] * Fmat[None])
        mean = torch.bmm(z0_mean[:, None], A.transpose(1, 2)).squeeze(1)
        cov = process_covariance(A) + A @ init_cov @ A.transpose(-1, -2)

    cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1e-6 * P_EQ.expand_as(cov)
    return mean, cov


def velocity_score_scale_from_cov(cov: torch.Tensor):
    """
    Return ell_t from the CLD paper for the per-dimension covariance block.

    For each dimension the covariance is [[S_xx, S_xv], [S_xv, S_vv]], and
    the velocity score can be written as -ell_t times a unit-scale target.
    """
    cov_xx = cov[..., 0, 0]
    cov_xv = cov[..., 0, D]
    cov_vv = cov[..., D, D]
    det = cov_xx * cov_vv - cov_xv.square()
    return torch.sqrt(cov_xx / det.clamp_min(1e-12))


def velocity_score_stats(t: torch.Tensor):
    x0 = torch.zeros(1 if t.ndim == 0 else t.numel(), D, device=device)
    _, cov = hsm_mean_cov(x0, t)
    return velocity_score_scale_from_cov(cov), cov[..., D, D]


def sample_forward_hsm(x0: torch.Tensor, t: torch.Tensor):
    """
    Sample z_t from p_t(z_t | x0), marginalizing v0 as in HSM.
    Also return the residual scaled velocity-noise target and score scale.
    """
    mean, cov = hsm_mean_cov(x0, t)
    L = torch.linalg.cholesky(cov)

    noise = torch.randn(x0.shape[0], ZDIM, device=device)
    if L.ndim == 2:
        zt = mean + noise @ L.T
    else:
        zt = mean + torch.bmm(L, noise[..., None]).squeeze(-1)

    # With the paper's Cholesky parameterization, the velocity-score target is
    # the standard Normal noise components that enter the velocity variables.
    ell = velocity_score_scale_from_cov(cov)
    cov_vv = cov[..., D, D]
    target_alpha_v = noise[:, D:]
    target_residual_alpha_v = target_alpha_v - zt[:, D:] / (
        ell[..., None] * cov_vv[..., None]
    )

    return zt, target_residual_alpha_v, ell


def exact_marginal_velocity_score(x: torch.Tensor, v: torch.Tensor, t: float) -> torch.Tensor:
    """
    Exact ∇_v log p_t(x, v) for the toy eight-Gaussian data distribution.

    This is useful as a paper/math sanity check: with this score, SSCS should
    recover the mixture. If it does, remaining ring artifacts come from the
    learned score model rather than from the reverse sampler.
    """
    t_tensor = torch.as_tensor(t, device=device)
    A = torch.matrix_exp(Fmat * t_tensor)
    z0_mean = torch.cat([data_centers(), torch.zeros(8, D, device=device)], dim=-1)
    means = z0_mean @ A.T

    init_cov = _initial_covariance(x0_var=data_std**2, v0_var=1.0)
    cov = process_covariance(A) + A @ init_cov @ A.T
    cov = 0.5 * (cov + cov.T) + 1e-6 * P_EQ
    precision = torch.linalg.inv(cov)

    z = torch.cat([x, v], dim=-1)
    diff = z[:, None, :] - means[None]
    mahalanobis = torch.einsum("nkd,df,nkf->nk", diff, precision, diff)
    weights = torch.softmax(-0.5 * mahalanobis, dim=1)
    component_scores = -torch.einsum("df,nkf->nkd", precision, diff)[:, :, D:]
    return (weights[..., None] * component_scores).sum(dim=1)


def exact_reverse_a_step(z: torch.Tensor, h: float):
    """
    Exact sample from the analytically solvable A part of the SSCS sampler.
    """
    h_tensor = torch.as_tensor(h, device=device)
    A = torch.matrix_exp(Fbar_amat * h_tensor)
    cov = process_covariance(A)
    L = torch.linalg.cholesky(cov)
    return z @ A.T + torch.randn_like(z) @ L.T


# -----------------------------
# Time embedding and score net
# -----------------------------


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t shape: [B, 1]
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half,
                device=t.device,
            )
        )
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityScoreNet(nn.Module):
    def __init__(self, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.temb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(ZDIM + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, D),
        )

    def forward(self, x: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """
        x: [B, D]
        v: [B, D]
        t: [B, 1]
        returns residual scaled velocity-noise prediction [B, D]
        """
        te = self.temb(t)
        inp = torch.cat([x, v, te], dim=-1)
        return self.net(inp)


model = VelocityScoreNet(hidden=256).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def reset_model(hidden_size: int = 256, time_dim: int = 64):
    global model, opt
    model = VelocityScoreNet(hidden=hidden_size, time_dim=time_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


# -----------------------------
# Training
# -----------------------------


def sample_time(time_sampling: str, shape=()):
    if time_sampling == "uniform":
        return torch.empty(shape, device=device).uniform_(eps_t, T)
    if time_sampling == "log":
        u = torch.rand(shape, device=device)
        return eps_t * (T / eps_t) ** u
    raise ValueError(f"Unknown time sampling {time_sampling!r}; use 'log' or 'uniform'.")


def train(
    num_train_steps: int = num_steps,
    train_batch_size: int = batch_size,
    time_sampling: str = "uniform",
    log_interval: int = 100,
):
    model.train()
    losses = []

    progress = tqdm(range(1, num_train_steps + 1), desc="training", unit="step")
    for step in progress:
        x0 = sample_data(train_batch_size)

        t_batch = sample_time(time_sampling, (train_batch_size, 1))

        zt, target_alpha_v, _ell = sample_forward_hsm(x0, t_batch.squeeze(-1))
        xt = zt[:, :D]
        vt = zt[:, D:]

        pred_alpha_v = model(xt, vt, t_batch)

        loss = F.mse_loss(pred_alpha_v, target_alpha_v)
        losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % log_interval == 0 or step == num_train_steps:
            progress.set_postfix(loss=f"{loss.item():.6f}")

    return losses


# -----------------------------
# Reverse sampling
# -----------------------------


@torch.no_grad()
def velocity_score(
    x: torch.Tensor,
    v: torch.Tensor,
    t: float,
    score_source: str = "model",
) -> torch.Tensor:
    if score_source == "oracle":
        return exact_marginal_velocity_score(x, v, t)
    if score_source != "model":
        raise ValueError(f"Unknown score source {score_source!r}; use 'model' or 'oracle'.")

    t_batch = torch.full((x.shape[0], 1), t, device=device)
    ell, cov_vv = velocity_score_stats(torch.as_tensor(t, device=device))
    alpha_v = x.new_tensor(1.0) * v / (ell * cov_vv) + model(x, v, t_batch)
    return -ell * alpha_v


@torch.no_grad()
def sample_reverse(
    n: int = 4096,
    num_reverse_steps: int = 500,
    sampler: str = "sscs",
    score_source: str = "model",
):
    """
    Reverse-time sampler for the CLD-like process.

    Forward:
        dx = beta v dt
        dv = beta (-x - gamma v) dt + sigma dW

    Reverse in positive reverse time tau:
        dx = -beta v dτ
        dv = beta (x + gamma v + 2 gamma score_v) dτ + sigma dWbar

    The default uses the paper's SSCS-style splitting. Euler-Maruyama is kept
    as a baseline because it is useful for seeing the failure mode.
    """
    model.eval()

    # Approximate terminal prior. For large T, forward CLD approximately forgets data.
    x = torch.randn(n, D, device=device)
    v = torch.randn(n, D, device=device)

    h = (T - eps_t) / num_reverse_steps

    progress = tqdm(range(num_reverse_steps), desc="sampling", unit="step")
    if sampler == "em":
        for k in progress:
            t_now = T - k * h
            score_v = velocity_score(x, v, t_now, score_source=score_source)

            noise = torch.randn_like(v)
            x_old, v_old = x, v

            # Reverse CLD dynamics with Euler-Maruyama.
            x = x_old + (-beta * v_old) * h
            v = v_old + beta * (x_old + gamma * v_old + 2.0 * gamma * score_v) * h
            v = v + sigma * math.sqrt(h) * noise
    elif sampler == "sscs":
        z = torch.cat([x, v], dim=-1)
        for k in progress:
            t_now = T - (k + 0.5) * h
            z = exact_reverse_a_step(z, 0.5 * h)

            x_mid = z[:, :D]
            v_mid = z[:, D:]
            score_v = velocity_score(x_mid, v_mid, t_now, score_source=score_source)

            z[:, D:] = v_mid + sigma**2 * (score_v + v_mid) * h
            z = exact_reverse_a_step(z, 0.5 * h)

        x = z[:, :D]
        v = z[:, D:]
    else:
        raise ValueError(f"Unknown sampler {sampler!r}; use 'sscs' or 'em'.")

    # Small deterministic denoising step from forward time eps_t to 0.
    x = x - beta * eps_t * v

    return x.cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and sample the toy CLD diffusion model.")
    parser.add_argument("--num-steps", type=int, default=num_steps)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--num-reverse-steps", type=int, default=500)
    parser.add_argument("--sampler", choices=("sscs", "em"), default="sscs")
    parser.add_argument("--score-source", choices=("model", "oracle"), default="model")
    parser.add_argument("--time-sampling", choices=("uniform", "log"), default="uniform")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-name", default="cld_diffusion_samples.png")
    parser.add_argument("--loss-output-name", default="cld_diffusion_loss.png")
    return parser.parse_args()


def main():
    args = parse_args()
    reset_model(hidden_size=args.hidden_size, time_dim=args.time_dim)

    if args.results_dir is None:
        args.results_dir = Path(
            f"results/cld_{args.score_source}_{args.sampler}_{args.time_sampling}_"
            f"{args.num_steps}steps_{args.num_reverse_steps}rev"
        )

    losses = []
    if args.score_source == "model":
        losses = train(
            num_train_steps=args.num_steps,
            train_batch_size=args.batch_size,
            time_sampling=args.time_sampling,
            log_interval=args.log_interval,
        )
    else:
        print("using exact oracle score; skipping neural-network training")

    samples = sample_reverse(
        n=args.num_samples,
        num_reverse_steps=args.num_reverse_steps,
        sampler=args.sampler,
        score_source=args.score_source,
    )

    # Optional quick plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = sample_data(args.num_samples).cpu()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / args.output_name
    loss_output_path = args.results_dir / args.loss_output_name

    plt.figure()
    plt.scatter(real[:, 0], real[:, 1], s=3, alpha=0.4, label="data")
    sample_label = "CLD samples" if args.score_source == "model" else "oracle CLD samples"
    plt.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.4, label=sample_label)
    plt.axis("equal")
    plt.legend()
    plt.title("Toy critically-damped Langevin diffusion")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved plot to {output_path}")

    if losses:
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel("training step")
        plt.ylabel("MSE loss")
        plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(loss_output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"saved loss plot to {loss_output_path}")


if __name__ == "__main__":
    main()
