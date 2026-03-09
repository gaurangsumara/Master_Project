import numpy as np
import arviz as az
import pymc as pm
import os
import pytensor.tensor as pt
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde

user_ids = np.array([0,0,1,1,0,2,2])
dt       = np.array([1,3,1,2,5,1,4])
n        = np.array([1,2,1,2,3,1,2])
omega    = np.array([0,1,0,1,1,0,1])
n_users  = 3
true_alpha = [0.6, 0.4, 0.8]
true_beta  = [0.7, 0.7, 0.3]
item_ids = np.array([0,1,0,1,2,0,2])
n_items  = 3

def hierarchical_memory_model(
    user_ids,
    item_ids,
    dt,
    n,
    omega,
    n_users,
    n_items
):

    with pm.Model() as model:

        # hyperpriors
        lambda_alpha = pm.Exponential("lambda_alpha", 1.0)
        lambda_item = pm.Exponential("lambda_item", 1.0)
        mu_beta_user = pm.Normal("mu_beta_user", 0, 1)
        sigma_beta_user = pm.Exponential("sigma_beta_user", 1)

        mu_beta_item = pm.Normal("mu_beta_item", 0, 1)
        sigma_beta_item = pm.Exponential("sigma_beta_item", 1)


        # non centered alpha

        alpha_user_raw = pm.Exponential("alpha_user_raw", 1, shape=n_users)

        alpha_user = pm.Deterministic(
            "alpha_user",
            alpha_user_raw / lambda_alpha
        )


        alpha_item_raw = pm.Exponential("alpha_item_raw", 1, shape=n_items)

        alpha_item = pm.Deterministic(
            "alpha_item",
            alpha_item_raw / lambda_item
        )


        # non-centered beta

        beta_user_raw = pm.Normal(
            "beta_user_raw",
            0,
            1,
            shape=n_users
        )

        beta_user = pm.Deterministic(

            "beta_user",

            pm.math.sigmoid(
                mu_beta_user
                + sigma_beta_user * beta_user_raw
            )
        )


        beta_item_raw = pm.Normal(
            "beta_item_raw",
            0,
            1,
            shape=n_items
        )

        beta_item = pm.Deterministic(

            "beta_item",

            pm.math.sigmoid(
                mu_beta_item
                + sigma_beta_item * beta_item_raw
            )
        )


        alpha_ui = (

            alpha_user[user_ids]
            + alpha_item[item_ids]

        ) / 2


        beta_ui = (

            beta_user[user_ids]
            + beta_item[item_ids]

        ) / 2


        p = pm.math.exp(

            -alpha_ui
            * pt.pow(1 - beta_ui, pt.maximum(n - 1, 0))
            * dt
        )


        p = pm.math.clip(p, 1e-6, 1 - 1e-6)


        pm.Bernoulli("obs", p=p, observed=omega)


    return model





#visualizations
#histogram for posterior distribution
def posterior_histogram(n_users, alpha_user_samples, beta_user_samples):

    for u in range(n_users):

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.hist(alpha_user_samples[:,u], bins=50)
        plt.title(f"Posterior alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.hist(beta_user_samples[:,u], bins=50)
        plt.title(f"Posterior beta_user[{u}]")

        plt.show()

#scatter plot 
def show_scatter_plot(n_users,alpha_user_samples, beta_user_samples):
    

    for u in range(n_users):

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.hist(alpha_user_samples[:,u], bins=50)
        plt.title(f"Posterior alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.hist(beta_user_samples[:,u], bins=50)
        plt.title(f"Posterior beta_user[{u}]")

        plt.show()

#posterior for all users
def alpha_user_poseterior(n_users, alpha_user_samples, beta_user_samples):
    for u in range(n_users):

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.hist(alpha_user_samples[:,u], bins=50)
        plt.title(f"Posterior alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.hist(beta_user_samples[:,u], bins=50)
        plt.title(f"Posterior beta_user[{u}]")

        plt.show()

#visualize posterior density
def posterior_density(alpha_user_samples, beta_user_samples):
    plt.figure(figsize=(6,5))

    plt.hexbin(
        alpha_user_samples[:,0],
        beta_user_samples[:,0],
        gridsize=40
    )

    plt.xlabel("alpha")
    plt.ylabel("beta")

    plt.title("Posterior density")

    plt.show()

#visualize density plot
def show_density_plot(n_users, alpha_user_samples,):
    for u in range(n_users):

        plt.figure(figsize=(10,4))

        kde = gaussian_kde(alpha_user_samples[:,u])
        x = np.linspace(min(alpha_user_samples[:,u]), max(alpha_user_samples[:,u]), 200)

        plt.plot(x, kde(x))
        plt.title(f"Density alpha_user[{u}]")

        plt.show()

#visualize alpha of all users
def comparision_between_users(n_users, alpha_user_samples):

    plt.figure(figsize=(8,6))

    for u in range(n_users):

        plt.hist(
            alpha_user_samples[:,u],
            bins=50,
            alpha=0.4,
            label=f"user {u}"
        )

        plt.legend()
        plt.title("Comparison between users")
    plt.show()

#posterior scatter plot for all users
def users_scatter_plot(n_users, alpha_user_samples, beta_user_samples):
    for u in range(n_users):

        plt.figure(figsize=(6,5))

        plt.scatter(
            alpha_user_samples[:,u],
            beta_user_samples[:,u],
            alpha=0.3
        )

        plt.xlabel("alpha")
        plt.ylabel("beta")

        plt.title(f"Posterior scatter user {u}")

        plt.show()

#trace plot for sampling behavior
def show_scatter_plot(n_users,alpha_user_samples, beta_user_samples):
    for u in range(n_users):

        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(alpha_user_samples[:,u])
        plt.title(f"Trace alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.plot(beta_user_samples[:,u])
        plt.title(f"Trace beta_user[{u}]")

        plt.show()

def trace_plot(n_users,alpha_user_samples, beta_user_samples):

    for u in range(n_users):

        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(alpha_user_samples[:,u])
        plt.title(f"Trace alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.plot(beta_user_samples[:,u])
        plt.title(f"Trace beta_user[{u}]")

    plt.show()

model = hierarchical_memory_model(
        user_ids=user_ids,
        item_ids=item_ids,
        dt=dt,
        n=n,
        omega=omega,
        n_users=n_users,
        n_items=n_items
    )

def inference(model, draws = 2000, tune= 2000):
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=0.95,
            return_inferencedata=True,
            max_tree_depth = 15
        )

    return trace

trace = inference(model)
alpha = trace.posterior["alpha_user"].values
beta  = trace.posterior["beta_user"].values

chains, draws, n_users = alpha.shape

alpha = alpha.reshape(-1, n_users)
beta  = beta.reshape(-1, n_users)

# posterior_histogram(n_users,alpha, beta)
posterior_density(alpha, beta)
show_density_plot(n_users,alpha)
trace_plot(n_users, alpha, beta)
comparision_between_users(n_users,alpha)
users_scatter_plot(n_users,alpha,beta)
