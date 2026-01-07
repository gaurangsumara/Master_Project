import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
# define the model
def base_model(user_ids, dt, n, omega, n_users):
    with pm.Model() as model:
        #conjugate priors
        lambda_alpha = pm.Gamma("lambda_alpha", alpha = 2.0 , beta = 1.0)
        beta_a = pm.Gamma("beta_a", alpha = 2.0, beta= 1.0)
        beta_b = pm.Gamma("beta_b", alpha = 2.0, beta= 1.0)

        #user parameters
        # alpha = pm.Exponential("alpha", lam=lambda_alpha, shape=n_users)
        # beta = pm.Beta("beta", alpha=beta_a, beta=beta_b, shape=n_users)
        log_alpha = pm.Normal("log_alpha", mu=0.0, sigma=1.0, shape=n_users)
        alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))

        beta_raw = pm.Normal("beta_raw", 0, 1, shape=n_users)
        beta = pm.Deterministic("beta", pm.math.sigmoid(beta_raw))


        #likelihood
        p = pm.math.exp(
            -alpha[user_ids]
            * (1 - beta[user_ids]) ** pt.maximum(n - 1, 0)
            * dt
        )

        p = pm.math.clip(p, 1e-6, 1 - 1e-6)

        pm.Bernoulli("omega", p=p, observed=omega)

    return model



def plot_user_posteriors(alpha_samples, beta_samples, true_alpha, true_beta):
    

    n_users = alpha_samples.shape[1]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(8, 6))

    for u in range(n_users):
        plt.scatter(
            alpha_samples[:, u],
            beta_samples[:, u],
            s=10,
            alpha=0.3,
            color=colors[u],
            label=f"User {u+1} samples"
        )

        # True parameters
        plt.scatter(
            true_alpha[u],
            true_beta[u],
            color=colors[u],
            marker="X",
            s=200,
            edgecolor="black",
            linewidth=2,
            label=f"User {u+1} true"
        )

        # Posterior mean (optional but recommended)
        plt.scatter(
            alpha_samples[:, u].mean(),
            beta_samples[:, u].mean(),
            color=colors[u],
            marker="o",
            s=120,
            edgecolor="black",
            linewidth=1.5
        )

    plt.xlabel("α (forgetting rate)")
    plt.ylabel("β (learning effect)")
    plt.title("Posterior samples per user with true parameters")
    plt.legend()
    plt.grid(True)
    plt.show()
