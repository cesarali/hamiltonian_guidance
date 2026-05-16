# toy_classifier_guidance_gmm.py
#
# Classifier guidance on a 2D Gaussian mixture.
#
# Each Gaussian component is one class.
# We use:
#   - exact unconditional score ∇ log p_t(x)
#   - exact classifier gradient ∇ log p_t(y | x)
#
# This demonstrates classifier guidance without neural networks.

import math
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Device and dtype
# ----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32


# ----------------------------
# Data distribution
# ----------------------------

# Four class-conditional Gaussians in R^2.
means = torch.tensor(
    [
        [-3.0, 0.0],
        [3.0, 0.0],
        [0.0, 3.0],
        [0.0, -3.0],
    ],
    device=device,
    dtype=dtype,
)

num_classes = means.shape[0]
dim = 2

data_std = 0.35
data_var = data_std**2


def sample_data(n):
    y = torch.randint(0, num_classes, (n,), device=device)
    x = means[y] + data_std * torch.randn(n, dim, device=device)
    return x, y


# ----------------------------
# Forward VP diffusion
# ----------------------------

# Forward SDE:
#   dx = -0.5 beta x dt + sqrt(beta) dW
#
# For constant beta:
#   x_t | x_0 ~ N(alpha_t x_0, sigma_t^2 I)
#
# where:
#   alpha_t = exp(-0.5 beta t)
#   sigma_t^2 = 1 - exp(-beta t)

beta = 2.0
T = 4.0


def alpha(t):
    return torch.exp(-0.5 * beta * t)


def sigma2(t):
    return 1.0 - torch.exp(-beta * t)


def component_params_at_t(t):
    """
    p_t(x | y=k) = N(alpha_t mu_k, var_t I)

    because x0 | y=k ~ N(mu_k, data_var I)
    and x_t = alpha_t x0 + noise.
    """
    a = alpha(t)
    s2 = sigma2(t)
    mean_t = a * means
    var_t = (a**2) * data_var + s2
    return mean_t, var_t


# ----------------------------
# Exact GMM scores and classifier
# ----------------------------


def log_normal_iso(x, mean, var):
    """
    x:    [B, 2]
    mean: [K, 2]
    var: scalar tensor

    returns log N(x; mean_k, var I): [B, K]
    """
    diff = x[:, None, :] - mean[None, :, :]
    sq = (diff**2).sum(dim=-1)
    log_norm = -0.5 * dim * torch.log(2.0 * torch.pi * var)
    return log_norm - 0.5 * sq / var


def class_log_probs_t(x, t):
    """
    Exact noisy classifier:
        log p_t(y=k | x_t=x)

    Since classes have uniform prior:
        p_t(y=k | x) ∝ p_t(x | y=k)
    """
    mean_t, var_t = component_params_at_t(t)
    log_px_given_y = log_normal_iso(x, mean_t, var_t)
    log_py_given_x = log_px_given_y - torch.logsumexp(
        log_px_given_y, dim=-1, keepdim=True
    )
    return log_py_given_x


def unconditional_score_t(x, t):
    """
    Exact score of the noisy mixture:
        ∇_x log p_t(x)

    For a mixture of isotropic Gaussians:
        score = sum_k r_k(x,t) * ∇ log N_k(x)
              = sum_k r_k * (mean_k(t) - x) / var_t
    """
    mean_t, var_t = component_params_at_t(t)
    log_px_given_y = log_normal_iso(x, mean_t, var_t)
    resp = torch.softmax(log_px_given_y, dim=-1)  # [B, K]

    component_scores = (mean_t[None, :, :] - x[:, None, :]) / var_t
    score = (resp[:, :, None] * component_scores).sum(dim=1)
    return score


def conditional_score_t(x, t, y):
    """
    Exact score of p_t(x | y):
        ∇_x log p_t(x | y)
        = (mean_y(t) - x) / var_t
    """
    mean_t, var_t = component_params_at_t(t)
    target_mean = mean_t[y]
    return (target_mean - x) / var_t


def classifier_guidance_grad_t(x, t, y):
    """
    Exact classifier guidance gradient:
        ∇_x log p_t(y | x)

    By Bayes:
        ∇ log p_t(y | x)
        = ∇ log p_t(x | y) - ∇ log p_t(x)
    """
    return conditional_score_t(x, t, y) - unconditional_score_t(x, t)


def guided_score_t(x, t, y, guidance_scale):
    """
    Classifier-guided score:
        s_guided = s_uncond + w ∇ log p_t(y | x)

    If w = 0: unconditional generation.
    If w = 1: exact conditional score for this GMM.
    If w > 1: over-guided, sharper / lower-diversity samples.
    """
    s_uncond = unconditional_score_t(x, t)
    grad_cls = classifier_guidance_grad_t(x, t, y)
    return s_uncond + guidance_scale * grad_cls


# ----------------------------
# Reverse SDE sampler
# ----------------------------


@torch.no_grad()
def sample_guided(
    n=2000,
    target_class=0,
    guidance_scale=1.0,
    num_steps=500,
):
    """
    Reverse-time VP SDE.

    Forward:
        dx = f(x,t) dt + g(t) dW
        f = -0.5 beta x
        g^2 = beta

    Reverse SDE:
        dx = [f(x,t) - g^2 score_t(x)] dt + g dW_bar

    We integrate from t=T down to t=0 using dt = -h:
        x <- x + drift * (-h) + sqrt(beta h) noise

    Equivalently:
        x <- x - drift*h + sqrt(beta h) noise
    """
    x = torch.randn(n, dim, device=device)
    y = torch.full((n,), target_class, device=device, dtype=torch.long)

    h = T / num_steps

    for i in range(num_steps):
        t_value = T - i * h
        t = torch.tensor(t_value, device=device, dtype=dtype)

        score = guided_score_t(x, t, y, guidance_scale)

        f = -0.5 * beta * x
        drift = f - beta * score

        noise = torch.randn_like(x) if i < num_steps - 1 else torch.zeros_like(x)

        # because we step backward in t
        x = x - drift * h + math.sqrt(beta * h) * noise

    return x.cpu()


# ----------------------------
# Plotting
# ----------------------------


def main():
    target_class = 0

    real_x, real_y = sample_data(4000)
    real_x = real_x.cpu()
    real_y = real_y.cpu()

    samples_w0 = sample_guided(
        n=3000,
        target_class=target_class,
        guidance_scale=0.0,
        num_steps=700,
    )

    samples_w1 = sample_guided(
        n=3000,
        target_class=target_class,
        guidance_scale=1.0,
        num_steps=700,
    )

    samples_w3 = sample_guided(
        n=3000,
        target_class=target_class,
        guidance_scale=3.0,
        num_steps=700,
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(samples_w0[:, 0], samples_w0[:, 1], s=4, alpha=0.4)
    plt.scatter(means[:, 0].cpu(), means[:, 1].cpu(), c="black", marker="x", s=80)
    plt.title("w = 0: unconditional")
    plt.axis("equal")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.subplot(1, 3, 2)
    plt.scatter(samples_w1[:, 0], samples_w1[:, 1], s=4, alpha=0.4)
    plt.scatter(means[:, 0].cpu(), means[:, 1].cpu(), c="black", marker="x", s=80)
    plt.title("w = 1: conditional class 0")
    plt.axis("equal")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.subplot(1, 3, 3)
    plt.scatter(samples_w3[:, 0], samples_w3[:, 1], s=4, alpha=0.4)
    plt.scatter(means[:, 0].cpu(), means[:, 1].cpu(), c="black", marker="x", s=80)
    plt.title("w = 3: strong guidance")
    plt.axis("equal")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
