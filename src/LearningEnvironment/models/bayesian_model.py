import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
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



def plot_user_posteriors_individual(alpha_samples, beta_samples, true_alpha, true_beta):
    

    n_users = alpha_samples.shape[1]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(8, 6))
    for u in range(n_users):
        a = alpha_samples[u]
        b = beta_samples[u]

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 2)

        # α marginal
        ax1 = fig.add_subplot(gs[0, 0])
        sns.kdeplot(a, ax=ax1, fill=True)
        ax1.set_title(f"User {u+1}: α posterior")
        ax1.set_xlabel("α")

        # β marginal
        ax2 = fig.add_subplot(gs[0, 1])
        sns.kdeplot(b, ax=ax2, fill=True)
        ax2.set_title(f"User {u+1}: β posterior")
        ax2.set_xlabel("β")

        # Joint posterior
        ax3 = fig.add_subplot(gs[1, :])
        sns.kdeplot(
            x=a, y=b,
            levels=5,
            fill=True,
            cmap="Blues",
            ax=ax3
        )
        ax3.scatter(a[::50], b[::50], s=5, alpha=0.3)
        ax3.set_xlabel("α")
        ax3.set_ylabel("β")
        ax3.set_title(f"User {u+1}: joint posterior")

        plt.tight_layout()
        plt.show()

        lambda_samples = a * (1 - b)
        sns.kdeplot(lambda_samples, fill=True)
        plt.title(f"User {u+1}: effective forgetting rate λ")
        plt.show()


def plot_user_posteriors(
    alpha_samples,
    beta_samples,
    true_alpha,
    true_beta
):
    n_users = alpha_samples.shape[1]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for u in range(n_users):
        # create a separate figure for each user
        plt.figure(figsize=(7, 6))

        # Posterior samples for user u
        plt.scatter(
            alpha_samples[:, u],
            beta_samples[:, u],
            s=10,
            alpha=0.3,
            color=colors[u],
            label="Posterior samples"
        )

        # True parameters (if known / simulated)
        plt.scatter(
            true_alpha[u],
            true_beta[u],
            color=colors[u],
            marker="X",
            s=200,
            edgecolor="black",
            linewidth=2,
            label="True parameters"
        )

        # Posterior mean
        plt.scatter(
            alpha_samples[:, u].mean(),
            beta_samples[:, u].mean(),
            color="black",
            marker="o",
            s=120,
            label="Posterior mean"
        )

        plt.xlabel("α (forgetting rate)")
        plt.ylabel("β (learning effect)")
        plt.title(f"User {u+1}: Posterior samples and true parameters")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


#individual alpha-beta posterior per user
def individual_alpha_beta_posterior(alpha_samples,beta_samples):
    n_users = alpha_samples.shape[-1]
    for u in range(n_users):
        a_u = alpha_samples[:, :, u].ravel()
        b_u = beta_samples[:, :, u].ravel()

    plt.figure(figsize=(6, 5))
    plt.scatter(a_u, b_u, s=8, alpha=0.25)
    plt.xlabel(r"$\alpha$ (forgetting rate)")
    plt.ylabel(r"$\beta$ (learning effect)")
    plt.title(f"Posterior samples for User {u+1}")
    plt.grid(True)
    plt.show()

#marginal distributions
def marginal_distributions(alpha_samples, beta_samples):
    n_users = alpha_samples.shape[-1]
    for u in range(n_users):
        a_u = alpha_samples[:, :, u].ravel()
        b_u = beta_samples[:, :, u].ravel()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(a_u, bins=50, density=True, alpha=0.7)
        axes[0].set_title(f"User {u+1}: α posterior")
        axes[0].set_xlabel(r"$\alpha$")
        axes[0].grid(True)

        axes[1].hist(b_u, bins=50, density=True, alpha=0.7)
        axes[1].set_title(f"User {u+1}: β posterior")
        axes[1].set_xlabel(r"$\beta$")
        axes[1].grid(True)

        plt.show()

# b1 vs b2 joint posterior plot

def b1_b2_joint_posterior(beta_samples):
    b1 = beta_samples[:, :, 0].ravel()
    b2 = beta_samples[:, :, 1].ravel()

    plt.figure(figsize=(6, 5))
    plt.scatter(b1, b2, s=8, alpha=0.25)
    plt.xlabel(r"$\beta_1$")
    plt.ylabel(r"$\beta_2$")
    plt.title("Joint posterior: β₁ vs β₂")
    plt.grid(True)
    plt.show()


