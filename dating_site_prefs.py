import random
import bisect

W_RANDOM, W_RANK = 0.5, 0.5

class RankGenerator:
    # Samples (man_score, woman_score) pairs using inverse transform sampling.

    def __init__(self):
        freqs = [
            [0.02661,0.00571,0.00249,0.00250,0.00436,0.00404,0.00539,0.00534,0.00360,0.01151],
            [0.00442,0.01155,0.00191,0.00152,0.00271,0.00253,0.00324,0.00310,0.00209,0.00611],
            [0.00251,0.00353,0.00575,0.00158,0.00248,0.00209,0.00277,0.00255,0.00192,0.00565],
            [0.00209,0.00210,0.00193,0.00455,0.00308,0.00223,0.00317,0.00307,0.00206,0.00642],
            [0.00230,0.00208,0.00239,0.00288,0.00906,0.00389,0.00565,0.00504,0.00360,0.01147],
            [0.00152,0.00117,0.00124,0.00138,0.00293,0.00552,0.00403,0.00431,0.00314,0.00962],
            [0.00137,0.00115,0.00132,0.00152,0.00317,0.00332,0.00872,0.00651,0.00513,0.01504],
            [0.00125,0.00076,0.00093,0.00102,0.00255,0.00235,0.00374,0.00835,0.00631,0.01828],
            [0.00095,0.00065,0.00084,0.00097,0.00178,0.00214,0.00277,0.00385,0.00814,0.02309],
            [0.00276,0.00217,0.00168,0.00150,0.00379,0.00350,0.00605,0.00883,0.01375,0.56255]
        ]

        cdf, pairs = [], []
        running = 0.0

        for i in range(10):
            for j in range(10):
                running += freqs[i][j]
                cdf.append(running)
                pairs.append((i + 1, j + 1))

        self.cdf = cdf
        self.pairs = pairs

    def get_scores(self):
        r = random.random()
        idx = bisect.bisect_left(self.cdf, r)
        return self.pairs[idx]

def make_preferences(score_row, rank_sums, N):
    # Converts score vector into a sorted preference list.

    def key_fn(j):
        return (
            score_row[j],
            W_RANK * (rank_sums[j] / N),
            W_RANDOM * (random.random() - 0.5),
        )

    order = list(range(N))
    order.sort(key=key_fn, reverse=True)
    return order

def generate_preferences(N, seed=None):
    # Generate preference lists for N men and N women, renumbered so
    # that man 0 and woman 0 are the MOST POPULAR.

    if seed is not None:
        random.seed(seed)

    dist = RankGenerator()

    # raw score matrices
    men_scores = [[0]*N for _ in range(N)]
    women_scores = [[0]*N for _ in range(N)]

    # popularity counters
    men_pop = [0]*N
    women_pop = [0]*N

    # generate compatibility scores
    for i in range(N):
        for j in range(N):
            m_score, w_score = dist.get_scores()
            men_scores[i][j] = m_score
            women_scores[j][i] = w_score
            men_pop[i] += w_score    # popularity = incoming scores
            women_pop[j] += m_score

    # initial preference lists
    men_prefs = [make_preferences(men_scores[i], women_pop, N) for i in range(N)]
    women_prefs = [make_preferences(women_scores[i], men_pop, N) for i in range(N)]
   
    # popularity ranks
    men_order = sorted(range(N), key=lambda x: men_pop[x], reverse=True)
    women_order = sorted(range(N), key=lambda x: women_pop[x], reverse=True)

    # mapping: old_id → new_id
    man_new_id = { old: new for new, old in enumerate(men_order) }
    woman_new_id = { old: new for new, old in enumerate(women_order) }

    # remap preference lists
    men_prefs_new = {}
    for old_m in range(N):
        new_m = man_new_id[old_m]
        men_prefs_new[new_m] = [woman_new_id[w] for w in men_prefs[old_m]]

    women_prefs_new = {}
    for old_w in range(N):
        new_w = woman_new_id[old_w]
        women_prefs_new[new_w] = [man_new_id[m] for m in women_prefs[old_w]]

    return men_prefs_new, women_prefs_new
