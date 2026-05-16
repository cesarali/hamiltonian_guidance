# toy_classifier_free_guidance_gmm.py
#
# Classifier-free guidance on a 2D Gaussian mixture.
#
# Each Gaussian component is one class.
# We train one conditional diffusion model with label dropout:
#
#   eps_theta(x_t, t, y)
#
# where y may be a real class label or a null/unconditional label.
#
# At sampling time:
#
#   eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
#
# This is the standard classifier-free guidance mechanism.

import argparse
import json
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

dim_x = 2
num_classes = 4
null_class = num_classes  # extra label for unconditional branch
num_labels = num_classes + 1

num_diffusion_steps = 200
batch_size = 512
num_train_steps = 20_000
lr = 2e-4
p_uncond = 0.15  # probability of dropping the class label


# -----------------------------
# Toy data: Gaussian mixture
# -----------------------------

means = torch.tensor(
    [
        [-3.0, 0.0],
        [3.0, 0.0],
        [0.0, 3.0],
        [0.0, -3.0],
    ],
    device=device,
)

data_std = 0.35


def sample_gmm(batch_size):
    """
    Returns:
        x0: [B, 2]
        y:  [B]
    """
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    x0 = means[y] + data_std * torch.randn(batch_size, dim_x, device=device)
    return x0, y


# -----------------------------
# DDPM noise schedule
# -----------------------------


def make_beta_schedule(T):
    """
    Simple linear beta schedule.
    """
    beta_start = 1e-4
    beta_end = 2e-2
    return torch.linspace(beta_start, beta_end, T, device=device)


betas = make_beta_schedule(num_diffusion_steps)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)


def reset_noise_schedule(diffusion_steps: int):
    global num_diffusion_steps, betas, alphas, alpha_bars
    global sqrt_alpha_bars, sqrt_one_minus_alpha_bars

    num_diffusion_steps = diffusion_steps
    betas = make_beta_schedule(num_diffusion_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)


def extract(values, t, x_shape):
    """
    values: [T]
    t:      [B]
    returns values[t] reshaped to [B, 1, ..., 1]
    """
    out = values.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def q_sample(x0, t, noise):
    """
    Forward diffusion:
        x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps
    """
    sqrt_ab = extract(sqrt_alpha_bars, t, x0.shape)
    sqrt_omab = extract(sqrt_one_minus_alpha_bars, t, x0.shape)
    return sqrt_ab * x0 + sqrt_omab * noise


# -----------------------------
# Embeddings
# -----------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B], integer timesteps
        returns: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device).float()
            / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


# -----------------------------
# Noise-prediction network
# -----------------------------


class CFGDenoiser(nn.Module):
    def __init__(
        self,
        dim_x=2,
        num_labels=5,
        time_dim=64,
        label_dim=32,
        hidden_dim=128,
    ):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.label_emb = nn.Embedding(num_labels, label_dim)

        self.net = nn.Sequential(
            nn.Linear(dim_x + time_dim + label_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim_x),
        )

    def forward(self, x_t, t, y):
        """
        x_t: [B, 2]
        t:   [B]
        y:   [B], class labels including null_class
        """
        te = self.time_emb(t)
        ye = self.label_emb(y)
        inp = torch.cat([x_t, te, ye], dim=-1)
        return self.net(inp)


model = CFGDenoiser(
    dim_x=dim_x,
    num_labels=num_labels,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def reset_model(
    hidden_dim: int = 128,
    time_dim: int = 64,
    label_dim: int = 32,
    learning_rate: float = lr,
):
    global model, optimizer

    model = CFGDenoiser(
        dim_x=dim_x,
        num_labels=num_labels,
        time_dim=time_dim,
        label_dim=label_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )


# -----------------------------
# Training
# -----------------------------


def train(
    train_steps: int = num_train_steps,
    train_batch_size: int = batch_size,
    drop_prob: float = p_uncond,
    log_interval: int = 100,
):
    model.train()
    losses = []

    progress = tqdm(range(1, train_steps + 1), desc="training", unit="step")
    for step in progress:
        x0, y = sample_gmm(train_batch_size)

        # Sample random diffusion time.
        t = torch.randint(
            0,
            num_diffusion_steps,
            (train_batch_size,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, noise)

        # Classifier-free training:
        # randomly replace y by the null/unconditional class.
        drop_mask = torch.rand(train_batch_size, device=device) < drop_prob
        y_train = y.clone()
        y_train[drop_mask] = null_class

        pred_noise = model(x_t, t, y_train)

        loss = F.mse_loss(pred_noise, noise)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % log_interval == 0 or step == train_steps:
            progress.set_postfix(loss=f"{loss.item():.6f}")

    return losses


# -----------------------------
# DDPM reverse sampling with CFG
# -----------------------------


@torch.no_grad()
def p_sample_cfg(x_t, t, target_class, guidance_scale):
    """
    One reverse DDPM step using classifier-free guidance.
    """
    B = x_t.shape[0]

    t_batch = torch.full(
        (B,),
        t,
        device=device,
        dtype=torch.long,
    )

    y_cond = torch.full(
        (B,),
        target_class,
        device=device,
        dtype=torch.long,
    )

    y_uncond = torch.full(
        (B,),
        null_class,
        device=device,
        dtype=torch.long,
    )

    eps_cond = model(x_t, t_batch, y_cond)
    eps_uncond = model(x_t, t_batch, y_uncond)

    # Classifier-free guidance formula.
    eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    # DDPM mean:
    # mu_theta = 1/sqrt(alpha_t) * (
    #     x_t - beta_t / sqrt(1-alpha_bar_t) * eps_theta
    # )
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_cfg
    )

    if t == 0:
        return mean

    noise = torch.randn_like(x_t)

    # Simple DDPM variance choice.
    # For a toy example, using beta_t is fine.
    sigma_t = torch.sqrt(beta_t)

    return mean + sigma_t * noise


@torch.no_grad()
def sample_cfg(
    n=2000,
    target_class=0,
    guidance_scale=1.0,
    show_progress=True,
):
    model.eval()

    x = torch.randn(n, dim_x, device=device)

    steps = reversed(range(num_diffusion_steps))
    if show_progress:
        steps = tqdm(
            steps,
            total=num_diffusion_steps,
            desc=f"sampling w={guidance_scale:g}",
            unit="step",
        )

    for t in steps:
        x = p_sample_cfg(
            x_t=x,
            t=t,
            target_class=target_class,
            guidance_scale=guidance_scale,
        )

    return x.cpu()


# -----------------------------
# Visualization
# -----------------------------


def class_sensitivity_metrics(samples_by_scale, target_class: int):
    means_cpu = means.detach().cpu()
    metrics = {}

    for scale, samples in samples_by_scale.items():
        distances = torch.cdist(samples, means_cpu)
        nearest = distances.argmin(dim=1)
        counts = torch.bincount(nearest, minlength=num_classes).float()
        fractions = counts / samples.shape[0]

        target_distance = torch.linalg.norm(
            samples - means_cpu[target_class][None, :],
            dim=1,
        )
        metrics[str(scale)] = {
            "nearest_class_fractions": [round(v, 4) for v in fractions.tolist()],
            "target_fraction": round(fractions[target_class].item(), 4),
            "mean_distance_to_target": round(target_distance.mean().item(), 4),
            "sample_mean": [round(v, 4) for v in samples.mean(dim=0).tolist()],
            "sample_std": [round(v, 4) for v in samples.std(dim=0).tolist()],
        }

    return metrics


def save_sample_plot(samples_by_scale, target_class: int, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_panels = len(samples_by_scale)
    plt.figure(figsize=(4 * num_panels, 4))

    for idx, (scale, samples) in enumerate(samples_by_scale.items(), start=1):
        plt.subplot(1, num_panels, idx)
        plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.4)
        plt.scatter(means[:, 0].cpu(), means[:, 1].cpu(), c="black", marker="x", s=80)
        plt.scatter(
            means[target_class, 0].cpu(),
            means[target_class, 1].cpu(),
            c="red",
            marker="x",
            s=110,
        )
        plt.title(f"guidance scale = {scale:g}\nclass {target_class}")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_loss_plot(losses, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.title("CFG denoiser training loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and sample the toy classifier-free guidance diffusion model."
    )
    parser.add_argument("--num-steps", type=int, default=num_train_steps)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--num-diffusion-steps", type=int, default=num_diffusion_steps)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument(
        "--guidance-scales",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 3.0],
    )
    parser.add_argument("--p-uncond", type=float, default=p_uncond)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--label-dim", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-name", default="cfg_diffusion_samples.png")
    parser.add_argument("--loss-output-name", default="cfg_diffusion_loss.png")
    parser.add_argument("--metrics-output-name", default="cfg_diffusion_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.target_class < 0 or args.target_class >= num_classes:
        raise ValueError(f"target class must be in [0, {num_classes - 1}]")
    if any(scale < 0 for scale in args.guidance_scales):
        raise ValueError("guidance scales must be non-negative")

    torch.manual_seed(args.seed)
    reset_noise_schedule(args.num_diffusion_steps)
    reset_model(
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        label_dim=args.label_dim,
        learning_rate=args.lr,
    )

    if args.results_dir is None:
        scales = "-".join(f"{scale:g}" for scale in args.guidance_scales)
        args.results_dir = Path(
            f"results/cfg_class{args.target_class}_{args.num_steps}steps_"
            f"{args.num_diffusion_steps}diff_w{scales}"
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)

    losses = train(
        train_steps=args.num_steps,
        train_batch_size=args.batch_size,
        drop_prob=args.p_uncond,
        log_interval=args.log_interval,
    )

    samples_by_scale = {}
    for scale in args.guidance_scales:
        samples_by_scale[scale] = sample_cfg(
            n=args.num_samples,
            target_class=args.target_class,
            guidance_scale=scale,
        )

    sample_output_path = args.results_dir / args.output_name
    loss_output_path = args.results_dir / args.loss_output_name
    metrics_output_path = args.results_dir / args.metrics_output_name

    save_sample_plot(samples_by_scale, args.target_class, sample_output_path)
    save_loss_plot(losses, loss_output_path)

    metrics = class_sensitivity_metrics(samples_by_scale, args.target_class)
    with metrics_output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"saved sample plot to {sample_output_path}")
    print(f"saved loss plot to {loss_output_path}")
    print(f"saved metrics to {metrics_output_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
