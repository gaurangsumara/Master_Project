import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor as pt
import arviz as az

def recall_probability(alpha, beta, n, dt):
    val = np.exp(-alpha * (1.0 - beta)**(max(0, n - 1)) * dt)
    # Clip to avoid log(0) later
    return np.clip(val, 1e-12, 1 - 1e-12)

def neg_log_likelihood(alpha, beta, data):

    total = 0.0
    for dt, n, omega in data:
        p = recall_probability(alpha, beta, n, dt)
        if omega == 1:
            total -= np.log(p)
        else:
            total -= np.log(1.0 - p)
    return float(total)

# lambda_alpha - the rate parameter of the exponential prior on alpha
# beta_a , beta_b - shape paramerteres for beta prior
def neg_log_posterior(alpha, beta, data, lambda_alpha=30.0, beta_a= 1.0, beta_b=1.0):
    nll = neg_log_likelihood(alpha, beta, data)
    #  alpha prior: exponential, -log p(alpha) = lambda * alpha + const
    alpha_prior = lambda_alpha * alpha

    # beta prior: Beta(a,b) -> -log p(beta) = -(a-1)ln(beta) - (b-1)ln(1-beta) + const
    eps = 1e-12
    beta_clipped = np.clip(beta, eps, 1.0-eps)
    beta_prior = 0.0

    if not (beta_a == 1.0 and beta_b == 1.0):
        beta_prior = -(beta_a - 1.0) * np.log(beta_clipped) - (beta_b - 1.0) * np.log(1.0 - beta_clipped)

    return float(nll + alpha_prior+beta_prior)

    

def map_grid_search(data, alpha_range=(0.01, 5.0), beta_range=(0.01, 0.99),
                    steps=200, lambda_alpha=30.0, beta_a=5.0, beta_b=2.0):

    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    betas  = np.linspace(beta_range[0], beta_range[1], steps)

    Z = np.zeros((steps, steps)) # rows = alpha, cols = beta
    best_nlp = np.inf
    best_alpha = None
    best_beta = None

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):

            val = neg_log_posterior(a, b, data, lambda_alpha=lambda_alpha,   beta_a=beta_a, beta_b=beta_b)
            Z[i, j] = val

            if val < best_nlp:
                best_nlp = val     
                best_alpha = a
                best_beta = b

    return best_alpha, best_beta, best_nlp, alphas, betas, Z


def plot_heatmap(alphas, betas, Z, best_alpha, best_beta):
    plt.figure(figsize=(10, 8))

    # Heatmap
    plt.imshow(
        Z,
        origin='lower',
        aspect='auto',
        extent=[betas.min(), betas.max(), alphas.min(), alphas.max()],
        cmap='viridis'
    )

    # Colorbar
    cbar = plt.colorbar()
    cbar.set_label("Negative Log Posterior (NLP)")

    # Mark MAP point
    plt.scatter(
        best_beta, best_alpha,
        color='red',
        s=60,
        marker='x',
        label=f"MAP (α={best_alpha:.3f}, β={best_beta:.3f})"
    )

    # Labels
    plt.xlabel("β (Repetition Effect)")
    plt.ylabel("α (Forgetting Rate)")
    plt.title("MAP Heatmap of Negative Log Posterior")

    plt.legend()
    plt.tight_layout()
    plt.show()





