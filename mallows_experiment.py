import algorithms
import utils
import pandas as pd 

def run():
    n, phis = 500, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    pred_gen_reps, algo_reps = 50, 50

    results_file = f"Mallows Results (n = {n}, pred reps = {pred_gen_reps}, algo reps = {algo_reps}).csv"
    df = pd.DataFrame(columns = ["phi", "rep", "DA_props", "PDA_props", "WDA_props", "stable", "PDA_instance_size", "WDA_instance_size"])

    for phi in phis:
        left_ranks, right_ranks = utils.generate_mallows_window_pred(n, phi, phi, pred_gen_reps)
        WDA_instance_size = sum([right_ranks[h] - left_ranks[h] + 1 for h in left_ranks.keys()])

        for i in range(algo_reps):
            res_prefs, hosp_prefs = utils.mallows_market(n, phi, phi)           
            _, DA_props_cnt = algorithms.DA(res_prefs, hosp_prefs)
            _, PDA_props_cnt, PDA_instance_size = algorithms.PDA_with_doubling(res_prefs, hosp_prefs, right_ranks, n//8)

            pruned_res_prefs = algorithms.window_prune(left_ranks, right_ranks, res_prefs, hosp_prefs)
            WDA_matching, WDA_props_cnt = algorithms.DA(pruned_res_prefs, hosp_prefs)
            is_stable = algorithms.check_stability(WDA_matching, res_prefs, hosp_prefs)

            new_row = {
                "phi": phi,
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

    utils.plot_mallows_results(results_file)

run()