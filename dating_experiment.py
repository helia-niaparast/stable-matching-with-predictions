import algorithms
import utils 
import pandas as pd
import dating_site_prefs 

def run():
    n_vals = [100, 300, 500, 700, 1000, 1300, 1600, 2000]
    pred_gen_reps, algo_reps = 50, 50    

    results_file = f"Dating Results (pred reps = {pred_gen_reps}, algo reps = {algo_reps}).csv"
    df = pd.DataFrame(columns = ["n", "rep", "DA_props", "PDA_props", "WDA_props", "stable", "PDA_instance_size", "WDA_instance_size"])

    for n in n_vals:
        left_ranks, right_ranks = utils.generate_dating_window_pred(n, pred_gen_reps)
        WDA_instance_size = sum([right_ranks[w] - left_ranks[w] + 1 for w in left_ranks.keys()])

        for i in range(algo_reps):
            men_prefs, women_prefs = dating_site_prefs.generate_preferences(n)          
            _, DA_props_cnt = algorithms.DA(men_prefs, women_prefs)
            _, PDA_props_cnt, PDA_instance_size = algorithms.PDA_with_doubling(men_prefs, women_prefs, right_ranks, n//8)

            pruned_men_prefs = algorithms.window_prune(left_ranks, right_ranks, men_prefs, women_prefs)
            WDA_matching, WDA_props_cnt = algorithms.DA(pruned_men_prefs, women_prefs)
            is_stable = algorithms.check_stability(WDA_matching, men_prefs, women_prefs)

            new_row = {
                "n": n,
                "rep": i,
                "DA_props": DA_props_cnt,
                "PDA_props": PDA_props_cnt,
                "WDA_props": WDA_props_cnt,
                "stable": is_stable,
                "PDA_instance_size": PDA_instance_size,
                "WDA_instance_size": WDA_instance_size,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index = True)
            df.to_csv(results_file, index = False)

    utils.plot_results(results_file)

run()