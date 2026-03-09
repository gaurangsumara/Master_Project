import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))
from LearningEnvironment.teacher.items import WordItem
from LearningEnvironment.teacher.base import Teacher
from LearningEnvironment.teacher.planning_contexts import EmptyPlanningContext, FixedHorizonContext, FixedLearnerContext
from LearningEnvironment.teacher.planners import RandomPlanner
from LearningEnvironment.agents.learners.exp_memory import ExpMemoryLearner
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



    # print("MAP α =", best_alpha)
    # print("MAP β =", best_beta)
    # print("MAP NLP =", best_nlp)


def run_experiment():
    interaction_data = default_session()
    map_grid_search(interaction_data)
    
    best_alpha, best_beta, best_nlp, alphas, betas, Z = map_grid_search(interaction_data,
        alpha_range=(0.01, 5.0),
        beta_range=(0.01, 0.99),
        steps=500
    )
    plot_heatmap(alphas, betas, Z, best_alpha, best_alpha)

run_experiment()