from prefsampling.ordinal import mallows
import random
import numpy as np 
import math
import algorithms 
import pandas as pd 
import matplotlib.pyplot as plt 
import dating_site_prefs 
import tiered_prefs 

def average(lst):
    return sum(lst)/len(lst)

def rank(element, list):
    for i, item in enumerate(list):
        if item == element:
            return i

def reverse_dict(dict):
    return {v:k for k,v in dict.items()}

def mallows_market(n, phi_r, phi_h):
    # IDs 0..n-1 for both sides; identity as central ranking 
    seed = random.randint(0, 10000)
    res_prefs = mallows(num_voters = n, num_candidates = n, phi = phi_r, seed = seed)
    hosp_prefs = mallows(num_voters = n, num_candidates = n, phi = phi_h, seed = seed + 1)
   
    res_prefs_dict = {i: np.array(res_prefs)[i].tolist() for i in range(n)}
    hosp_prefs_dict = {i: np.array(hosp_prefs)[i].tolist() for i in range(n)}

    return res_prefs_dict, hosp_prefs_dict

def generate_mallows_window_pred(n, phi_r, phi_h, reps):
    hosp_ranks = {h:[] for h in range(n)}
    for _ in range(reps):
        resident_prefs, hospital_prefs = mallows_market(n, phi_r, phi_h)
        matching, _ = algorithms.DA(resident_prefs, hospital_prefs) #{r:h}
        reverse_matching = reverse_dict(matching) #{h:r}
        for h,r in reverse_matching.items():
            hosp_ranks[h].append(rank(r, hospital_prefs[h]))
    
    stdevs = {h: np.std(np.array(hosp_ranks[h])) for h in range(n)}

    right_ranks = {h: min(math.ceil(max(hosp_ranks[h]) + 3*stdevs[h]), n-1) for h in range(n)}
    left_ranks = {h: max(math.floor(min(hosp_ranks[h]) - 3*stdevs[h]), 0) for h in range(n)}

    return left_ranks, right_ranks

def generate_dating_window_pred(n, reps):
    women_ranks = {w:[] for w in range(n)}
    for _ in range(reps):
        man_prefs, woman_prefs = dating_site_prefs.generate_preferences(n)
        matching, _ = algorithms.DA(man_prefs, woman_prefs) #{m:w}
        reverse_matching = reverse_dict(matching) #{w:m}
        for w,m in reverse_matching.items():
            women_ranks[w].append(rank(m, woman_prefs[w]))
    
    stdevs = {w: np.std(np.array(women_ranks[w])) for w in range(n)}

    right_ranks = {w: min(math.ceil(max(women_ranks[w]) + 3*stdevs[w]), n-1) for w in range(n)}
    left_ranks = {w: max(math.floor(min(women_ranks[w]) - 3*stdevs[w]), 0) for w in range(n)}

    return left_ranks, right_ranks

def generate_tiered_window_pred(n, alpha, beta, scores_m, scores_w, reps):
    women_ranks = {w:[] for w in range(n)}
    for _ in range(reps):
        men_prefs, women_prefs = tiered_prefs.generate_all_preferences(n, alpha, beta, scores_m, scores_w)
        matching, _ = algorithms.DA(men_prefs, women_prefs) #{m:w}
        reverse_matching = reverse_dict(matching) #{w:m}
        for w,m in reverse_matching.items():
            women_ranks[w].append(rank(m, women_prefs[w]))
    
    stdevs = {w: np.std(np.array(women_ranks[w])) for w in range(n)}

    right_ranks = {w: min(math.ceil(max(women_ranks[w]) + 3*stdevs[w]), n-1) for w in range(n)}
    left_ranks = {w: max(math.floor(min(women_ranks[w]) - 3*stdevs[w]), 0) for w in range(n)}

    return left_ranks, right_ranks

def plot_results(filename):
    df = pd.read_csv(filename)

    grouped = df.groupby("n")

    stats = grouped.agg({
        "DA_props": ['mean','std'],
        "PDA_props": ['mean','std'],
        "WDA_props": ['mean','std'],
        "PDA_instance_size": ['mean', 'std'],
        "WDA_instance_size": ['mean', 'std'],
        "stable": 'mean'
    }).reset_index().sort_values("n")

    n_vals = stats["n"]
    DA_means = stats[("DA_props","mean")]
    DA_stds = stats[("DA_props","std")]
    PDA_means = stats[("PDA_props","mean")]
    PDA_stds  = stats[("PDA_props","std")]
    WDA_means = stats[("WDA_props","mean")]
    WDA_stds  = stats[("WDA_props","std")]
    PDA_prune_means = stats[("PDA_instance_size", "mean")]
    PDA_prune_stds = stats[("PDA_instance_size", "std")]
    WDA_prune_means = stats[("WDA_instance_size", "mean")]
    WDA_prune_stds = stats[("WDA_instance_size", "std")]
    stable = stats[("stable", "mean")]

    plt.figure(figsize = (8, 5))
    plt.plot(n_vals, DA_means, label = "DA", color = "blue", marker = "o")
    plt.fill_between(n_vals, DA_means - DA_stds, DA_means + DA_stds, color = "blue", alpha = 0.2)
    plt.plot(n_vals, PDA_means, label = "PDA", color = "orange", marker = "s")
    plt.fill_between(n_vals, PDA_means - PDA_stds, PDA_means + PDA_stds, color = "orange", alpha = 0.2)
    plt.plot(n_vals, WDA_means, label = "WDA", color = "green", marker = "v")
    plt.fill_between(n_vals, WDA_means - WDA_stds, WDA_means + WDA_stds, color = "green", alpha = 0.2)

    for x, y, num in zip(n_vals, WDA_means, stable):
        plt.text(x, y + 0.1 * (max(WDA_means)), f"{num * 100:.1f}%", fontsize = 12, color = "darkgreen", ha = "center", va = "bottom", zorder = 10)

    fontsize = 14
    plt.xlabel(r"$n$", fontsize = fontsize)
    plt.ylabel("Number of Proposals", fontsize = fontsize)
    plt.legend(fontsize = fontsize)
    plt.grid(True, linestyle = "--", alpha = 0.6)
    plt.tight_layout()
    plt.savefig(filename[:-3] + "pdf")

    plt.figure(figsize = (8, 5))

    plt.plot(n_vals, PDA_prune_means, label = "PDA", color = "orange", marker = "s")
    plt.fill_between(n_vals, PDA_prune_means - PDA_prune_stds, PDA_prune_means + PDA_prune_stds, color = "orange", alpha = 0.2)
    plt.plot(n_vals, WDA_prune_means, label = "WDA", color = "green", marker = "v")
    plt.fill_between(n_vals, WDA_prune_means - WDA_prune_stds, WDA_prune_means + WDA_prune_stds, color = "green", alpha = 0.2)

    plt.xlabel(r"$n$", fontsize = fontsize)
    plt.ylabel("Instance Size", fontsize = fontsize)
    plt.legend(fontsize = fontsize)
    plt.grid(True, linestyle = "--", alpha = 0.6)
    plt.tight_layout()
    plt.savefig(filename[:-4] + " - instance size.pdf")

def plot_mallows_results(filename):
    df = pd.read_csv(filename)

    grouped = df.groupby("phi")

    stats = grouped.agg({
        "DA_props": ['mean','std'],
        "PDA_props": ['mean','std'],
        "WDA_props": ['mean','std'],
        "PDA_instance_size": ['mean', 'std'],
        "WDA_instance_size": ['mean', 'std'],
        "stable": 'mean'
    }).reset_index().sort_values("phi")

    phi_vals = stats["phi"]
    DA_means = stats[("DA_props","mean")]
    DA_stds = stats[("DA_props","std")]
    PDA_means = stats[("PDA_props","mean")]
    PDA_stds  = stats[("PDA_props","std")]
    WDA_means = stats[("WDA_props","mean")]
    WDA_stds  = stats[("WDA_props","std")]
    PDA_prune_means = stats[("PDA_instance_size", "mean")]
    PDA_prune_stds = stats[("PDA_instance_size", "std")]
    WDA_prune_means = stats[("WDA_instance_size", "mean")]
    WDA_prune_stds = stats[("WDA_instance_size", "std")]
    stable = stats[("stable", "mean")]

    plt.figure(figsize = (8, 5))
    plt.plot(phi_vals, DA_means, label = "DA", color = "blue", marker = "o")
    plt.fill_between(phi_vals, DA_means - DA_stds, DA_means + DA_stds, color = "blue", alpha = 0.2)
    plt.plot(phi_vals, PDA_means, label = "PDA", color = "orange", marker = "s")
    plt.fill_between(phi_vals, PDA_means - PDA_stds, PDA_means + PDA_stds, color = "orange", alpha = 0.2)
    plt.plot(phi_vals, WDA_means, label = "WDA", color = "green", marker = "v")
    plt.fill_between(phi_vals, WDA_means - WDA_stds, WDA_means + WDA_stds, color = "green", alpha = 0.2)

    for x, y, num in zip(phi_vals, WDA_means, stable):
        plt.text(x, y + 0.1 * (max(WDA_means)), f"{num * 100:.1f}%", fontsize = 12, color = "darkgreen", ha = "center", va = "bottom", zorder = 10)

    fontsize = 14
    plt.xlabel(r"$\phi$", fontsize = fontsize)
    plt.ylabel("Number of Proposals", fontsize = fontsize)
    plt.legend(fontsize = fontsize)
    plt.grid(True, linestyle = "--", alpha = 0.6)
    plt.tight_layout()
    plt.savefig(filename[:-3] + "pdf")

    plt.figure(figsize = (8, 5))

    plt.plot(phi_vals, PDA_prune_means, label = "PDA", color = "orange", marker = "s")
    plt.fill_between(phi_vals, PDA_prune_means - PDA_prune_stds, PDA_prune_means + PDA_prune_stds, color = "orange", alpha = 0.2)
    plt.plot(phi_vals, WDA_prune_means, label = "WDA", color = "green", marker = "v")
    plt.fill_between(phi_vals, WDA_prune_means - WDA_prune_stds, WDA_prune_means + WDA_prune_stds, color = "green", alpha = 0.2)

    plt.xlabel(r"$\phi$", fontsize = fontsize)
    plt.ylabel("Instance Size", fontsize = fontsize)
    plt.legend(fontsize = fontsize)
    plt.grid(True, linestyle = "--", alpha = 0.6)
    plt.tight_layout()
    plt.savefig(filename[:-4] + " - instance size.pdf")