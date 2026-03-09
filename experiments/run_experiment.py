import sys
import os
import numpy as np
import arviz as az
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from LearningEnvironment.models.bayesian_model import base_model
from LearningEnvironment.models.inference import inference
from LearningEnvironment.models.hierarchical_bayesian_model import hierarchical_memory_model,show_density_plot
from LearningEnvironment.analysis.posterior import extract_user_parameters

def generate_data():
    user_ids = np.array([0,0,1,1,0,2,2])
    dt       = np.array([1,3,1,2,5,1,4])
    n        = np.array([1,2,1,2,3,1,2])
    omega    = np.array([0,1,0,1,1,0,1])
    n_users  = 3
    true_alpha = [0.6, 0.4, 0.8]
    true_beta  = [0.7, 0.7, 0.3]
    item_ids = np.array([0,1,0,1,2,0,2])
    n_items  = 3
    return user_ids,item_ids,dt,n,omega,n_users,true_alpha,true_beta,n_items

def run_experiment():
    print("Generating Data...")
    user_ids,item_ids,dt, n, omega, n_users, true_alpha, true_beta, n_items = generate_data()

    print("Running Experiment...")
    model = hierarchical_memory_model(
        user_ids=user_ids,
        item_ids=item_ids,
        dt=dt,
        n=n,
        omega=omega,
        n_users=n_users,
        n_items=n_items
    )
    print("Initializing NUTS...")
    trace = inference(model)
    alpha_samples, beta_samples = extract_user_parameters(trace)
    #you can add other visualization functions calls here, 
    # refer to hierarchical_bayesian_model.py file for more information.
    show_density_plot(n_users,alpha_samples)
    print("Saving results...")

    os.makedirs("experiments/results", exist_ok=True)

    az.to_netcdf(

        trace,

        "experiments/results/hierarchical_trace.nc"

    )


    print("Experiment complete.")


    print("\nPosterior summary:\n")

    print(

        az.summary(

            trace,

            var_names=[

                "alpha_user",

                "beta_user",

                "alpha_item",

                "beta_item"

            ]

        )

    )
   
# Main entry point


if __name__ == "__main__":

    run_experiment()


# model = base_model(user_ids, dt, n, omega, n_users)
# trace = inference(model)

# alpha_samples = trace.posterior["alpha"].stack(sample=("chain", "draw")).values
# beta_samples  = trace.posterior["beta"].stack(sample=("chain", "draw")).values
# alpha_samples = trace.posterior["alpha"].values
# beta_samples  = trace.posterior["beta"].values
# shape: (chains, draws, users)
# print(alpha_samples)

# alpha_flat = alpha_samples.reshape(-1, alpha_samples.shape[-1])
# beta_flat  = beta_samples.reshape(-1, beta_samples.shape[-1])

# print(alpha_flat)
# print(trace.posterior["alpha"].mean(dim=("chain", "draw")))
# print(trace.posterior["beta"].mean(dim=("chain", "draw")))

# az.plot_trace(trace, var_names=["alpha", "beta"])
# az.summary(trace, var_names=["alpha", "beta"])
# alpha_flat = alpha_samples.reshape(-1, alpha_samples.shape[-1])
# beta_flat  = beta_samples.reshape(-1, beta_samples.shape[-1])


# plot_user_posteriors(
#     alpha_flat,
#     beta_flat,
#     true_alpha=true_alpha,
#     true_beta=true_beta
# )

# plot_user_posteriors_individual(
#     alpha_flat,
#     beta_flat,
#     true_alpha=true_alpha,
#     true_beta=true_beta
# )
# marginal_distributions(alpha_samples,beta_samples)
# b1_b2_joint_posterior(beta_samples)
# def trace_plots(alpha_samples):
#     n_users = alpha_samples.shape[-1]
#     for u in range(n_users):
#         az.plot_trace(trace, var_names=["alpha", "beta"], coords={"alpha_dim_0": u})
#         plt.suptitle(f"Trace plots for User {u+1}", y=1.02)
#         plt.show()

# individual_alpha_beta_posterior(alpha_samples,beta_samples)
# trace_plots(alpha_samples)



# lambda_samples = trace.posterior["lambda_alpha"].values.flatten()

# alpha_user_samples = trace.posterior["alpha_user"].values

# # flatten chains and draws
# alpha_user_samples = alpha_user_samples.reshape(-1, alpha_user_samples.shape[-1])

# # plot basic 
# plt.figure(figsize=(8,6))

# for u in range(alpha_user_samples.shape[1]):
#     plt.scatter(
#         lambda_samples,
#         alpha_user_samples[:,u],
#         alpha=0.2,
#         s=5
#     )

# plt.xlabel("lambda_alpha")
# plt.ylabel("alpha_user")
# plt.title("Funnel Posterior Visualization")
# plt.show()


# # plot with users
# az.plot_pair(
#     trace,
#     var_names=["lambda_alpha","alpha_user"],
#     kind="scatter",
#     marginals=True
# )
# plt.show()

# div = trace.sample_stats["diverging"].values.flatten()

# plt.scatter(
#     lambda_samples,
#     alpha_user_samples[:,0],
#     c=div,
#     cmap="coolwarm",
#     alpha=0.6
# )

# plt.colorbar(label="Divergence")
# plt.xlabel("lambda")
# plt.ylabel("alpha_user")
# plt.title("Divergences in Funnel")
# plt.show()

# plt.figure(figsize=(8,6))


# log scaled
# plt.scatter(
#     np.log(lambda_samples),
#     np.log(alpha_user_samples[:,0]),
#     alpha=0.2,
#     s=5
# )

# plt.xlabel("log lambda")
# plt.ylabel("log alpha_user[0]")
# plt.title("Log Funnel")
# plt.show()
# plt.scatter(lambda_samples, alpha_user_samples[:,0])


#non centred priors model









