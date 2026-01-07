from teacher.items import WordItem
from teacher.base import Teacher
from teacher.planning_contexts import EmptyPlanningContext, FixedHorizonContext, FixedLearnerContext
from teacher.planners import RandomPlanner
from learners.exp_memory import ExpMemoryLearner
from Psychologist.bayesian_model import base_model,plot_user_posteriors
from Psychologist.inference import inference
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

def default_session(steps = 10):
    material = [WordItem("dog", "hund"), 
                WordItem("cat", "katze"),
                WordItem("essen","eat"),
                WordItem("triken","drink"),
                WordItem("Pferd","Horse"),
                WordItem("kochen","cook"),
                WordItem("tanzen","dance"),
                WordItem("wasser","water"),
                WordItem("milch","milk"),
                WordItem("zug","train"),
                # WordItem("fahren","drive"),
                # WordItem("singen","sing"),
                # WordItem("tragen","wear"),
                # WordItem("blumen","flowers"),
                # WordItem("kleidung","clothes")
                ]



    #context = EmptyPlanningContext()
    context = FixedLearnerContext(ExpMemoryLearner(0.4, 1))
    planner = RandomPlanner()
    teacher = Teacher(material, planner, context)

    learner = ExpMemoryLearner(.05, .6)
    last_seen = {item.get_question(): None for item in material}
    n_occurrences = {item.get_question(): 0 for item in material}
    interaction_data = []

    for t in range(10):
        item = teacher.choose_item(t)
        question = item.get_question()
        reply = learner.reply(item.get_question(), t)
        learner.learn(item, t)
        teacher.gets_answer(item, reply, t)
        omega = 1 if reply is not None else 0
        prev_time = last_seen[question]
        dt = t - prev_time if prev_time is not None else 1
        n_occurrences[question] += 1
        interaction_data.append((dt, n_occurrences[question], omega))
        last_seen[question] = t
        print(f"Q: {question} | A: {reply} | ω={omega} | n={n_occurrences[question]} | Δt={dt}")

    return interaction_data

# if __name__ == "__main__":
#     data = default_session()  

#     best_alpha, best_beta, best_nlp, alphas, betas, Z = map_grid_search(
#         data,
#         alpha_range=(0.01, 5.0),
#         beta_range=(0.01, 0.99),
#         steps=500
#     )

#     print("MAP α =", best_alpha)
#     print("MAP β =", best_beta)
#     print("MAP NLP =", best_nlp)

user_ids = np.array([0,0,1,1,0,2,2])
dt       = np.array([1,3,1,2,5,1,4])
n        = np.array([1,2,1,2,3,1,2])
omega    = np.array([0,1,0,1,1,0,1])
n_users  = 3
true_alpha = [0.6, 0.4, 0.8]
true_beta  = [0.7, 0.6, 0.4]


model = base_model(user_ids, dt, n, omega, n_users)
trace = inference(model)



alpha_samples = trace.posterior["alpha"].values
beta_samples  = trace.posterior["beta"].values

# shape: (chains, draws, users)
# print(alpha_samples)

alpha_flat = alpha_samples.reshape(-1, alpha_samples.shape[-1])
beta_flat  = beta_samples.reshape(-1, beta_samples.shape[-1])

# print(alpha_flat)
print(trace.posterior["alpha"].mean(dim=("chain", "draw")))
print(trace.posterior["beta"].mean(dim=("chain", "draw")))

# az.plot_trace(trace, var_names=["alpha", "beta"])
# az.summary(trace, var_names=["alpha", "beta"])
# alpha_flat = alpha_samples.reshape(-1, alpha_samples.shape[-1])
# beta_flat  = beta_samples.reshape(-1, beta_samples.shape[-1])
plot_user_posteriors(
    alpha_flat,
    beta_flat,
    true_alpha=true_alpha,
    true_beta=true_beta
)
