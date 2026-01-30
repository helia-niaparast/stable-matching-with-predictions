import numpy as np

def generate_scores(n, fractions, scores):
    agents = []
    agent_scores = []
    for frac, score in zip(fractions, scores):
        num_agents = int(round(frac * n))
        agents.extend(range(len(agents), len(agents) + num_agents))
        agent_scores.extend([score] * num_agents)

    while len(agents) < n:
        agents.append(len(agents))
        agent_scores.append(scores[-1])
    
    return np.array(agents), np.array(agent_scores)

def generate_preference_list(agent_scores_other):
    remaining = list(range(len(agent_scores_other)))
    remaining_scores = agent_scores_other.copy()
    pref_list = []
    
    for _ in range(len(agent_scores_other)):
        probs = remaining_scores / remaining_scores.sum()
        choice = np.random.choice(remaining, p=probs)
        pref_list.append(choice)
        idx = remaining.index(choice)
        remaining.pop(idx)
        remaining_scores = np.delete(remaining_scores, idx)
    
    return pref_list

def generate_all_preferences(n, alpha, beta, scores_m, scores_w):
    _, men_scores = generate_scores(n, alpha, scores_m)
    _, women_scores = generate_scores(n, beta, scores_w)
    
    men_prefs = {i:generate_preference_list(women_scores) for i in range(n)}
    women_prefs = {i:generate_preference_list(men_scores) for i in range(n)}
    
    return men_prefs, women_prefs