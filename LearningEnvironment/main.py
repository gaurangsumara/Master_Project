from teacher.items import WordItem
from teacher.base import Teacher
from teacher.planning_contexts import EmptyPlanningContext, FixedHorizonContext, FixedLearnerContext
from teacher.planners import RandomPlanner
from learners.exp_memory import ExpMemoryLearner
from psychologist import plot_heatmap,mle_grid_search,recall_probability
import matplotlib.pyplot as plt
import numpy as np

def default_session(steps = 100):
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
                WordItem("fahren","drive"),
                WordItem("singen","sing"),
                WordItem("tragen","wear"),
                WordItem("blumen","flowers"),
                WordItem("kleidung","clothes")
                ]



    #context = EmptyPlanningContext()
    context = FixedLearnerContext(ExpMemoryLearner(0.4, 1))
    planner = RandomPlanner()
    teacher = Teacher(material, planner, context)

    learner = ExpMemoryLearner(.4, .1)
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

        n_occurrences[question] += 1
        interaction_data.append((dt, n_occurrences[question], omega))
        print(f"Q: {question} | A: {reply} | ω={omega} | n={n_occurrences[question]} | Δt={dt}")

    return interaction_data

def main():
    # run a session (change num_steps to collect more data)
    data = default_session(steps=200)

   
    best_alpha, best_beta, best_nll, alphas, betas, Z = mle_grid_search(
        data, alpha_range=(0.01, 50.0), beta_range=(0.01, 0.99), steps=200)

    print(f"\nGlobal MLE (grid): α={best_alpha:.4f}, β={best_beta:.4f}, NLL={best_nll:.4f}")
    plot_heatmap(data, best_alpha, best_beta, alpha_range=(0.01, 50.0), beta_range=(0.01, 0.99), steps=200)

if __name__ == "__main__":
    main()