import numpy as np
import matplotlib.pyplot as plt


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



def mle_grid_search(data, alpha_range=(0.01, 50.0), beta_range=(0.01, 0.99), steps=200):
    
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    betas  = np.linspace(beta_range[0], beta_range[1], steps)

    Z = np.zeros((steps, steps), dtype=float)  # rows = alpha, cols = beta

    best_nll = np.inf
    best_alpha = None
    best_beta = None

    # compute grid, vectorized over data inside loops
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            Z[i, j] = neg_log_likelihood(a, b, data)
            if Z[i, j] < best_nll:
                best_nll = Z[i, j]
                best_alpha = a
                best_beta = b

    return best_alpha, best_beta, best_nll, alphas, betas, Z



def plot_heatmap(data, best_alpha=None, best_beta=None,
                 alpha_range=(0.01, 5.0), beta_range=(0.01, 0.99), steps=200,
                 cmap='turbo'):


    best_a, best_b, best_nll, alphas, betas, Z = mle_grid_search(
        data, alpha_range=alpha_range, beta_range=beta_range, steps=steps)

    plt.figure(figsize=(9,6))
    # imshow expects matrix with first axis = Y (alpha), second = X (beta)
    extent = [betas.min(), betas.max(), alphas.min(), alphas.max()]
    img = plt.imshow(Z,
                     origin='lower',
                     extent=extent,
                     aspect='auto',
                     cmap=cmap)

    plt.colorbar(img, label='Negative Log-Likelihood')
    plt.xlabel('β (learning effect)')
    plt.ylabel('α (forgetting rate)')
    plt.title('NLL heatmap (grid search)')

    # Mark grid-search global minimum
    plt.scatter(best_a and best_b and [best_b] or [best_b], best_a, color='white',
                edgecolor='black', s=80, label=f'Grid MLE (α={best_a:.3f}, β={best_b:.3f})')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return best_a, best_b, best_nll, alphas, betas, Z
