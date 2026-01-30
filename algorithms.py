import utils
import copy

def DA(proposers_prefs, receivers_prefs, cap = 1e9):
    proposers = list(proposers_prefs.keys())
    receivers = list(receivers_prefs.keys())

    receiver_rank = {r: {p: i for i, p in enumerate(receivers_prefs[r])} for r in receivers}

    free_proposers = set(proposers)
    proposals = {p: 0 for p in proposers}
    current_matches = {}

    num_proposals = 0
    while free_proposers:
        p = free_proposers.pop()
        p_prefs = proposers_prefs[p]
        if proposals[p] == len(p_prefs):
            continue
        
        r = p_prefs[proposals[p]]
        proposals[p] += 1
        num_proposals += 1
        
        if r not in current_matches:
            current_matches[r] = p
        else:
            current_p = current_matches[r]
            if receiver_rank[r][p] < receiver_rank[r][current_p]:
                current_matches[r] = p
                free_proposers.add(current_p)
            else:
                free_proposers.add(p)
        
        if num_proposals == cap:
            break

    return {p: r for r, p in current_matches.items()}, num_proposals

def prefix_prune(matching, proposers_prefs, receivers_prefs):
    pruned_prefs = {}
    for p, p_prefs in proposers_prefs.items():
        pruned_list = []
        for r in p_prefs:
            r_prefs = receivers_prefs[r]
            if utils.rank(p, r_prefs) <= utils.rank(matching[r], r_prefs):
                pruned_list.append(r)
        pruned_prefs[p] = pruned_list
    
    return pruned_prefs

def window_prune(left_ranks, right_ranks, proposers_prefs, receivers_prefs):
    pruned_prefs = {}
    for p, p_prefs in proposers_prefs.items():
        pruned_list = []
        for r in p_prefs:
            if left_ranks[r] <= utils.rank(p, receivers_prefs[r]) <= right_ranks[r]:
                pruned_list.append(r)
        pruned_prefs[p] = pruned_list
    
    return pruned_prefs

def PDA_with_doubling(proposers_prefs, receivers_prefs, predicted_ranks, initial_stretch):
    size = 0
    n = len(proposers_prefs.items())

    updated_predicted_ranks = copy.deepcopy(predicted_ranks)
    updated_predicted_matching = {r:receivers_prefs[r][match_rank] for r, match_rank in updated_predicted_ranks.items()}
    matching, num_proposals = PDA_without_doubling(proposers_prefs, receivers_prefs, updated_predicted_matching)

    size += sum(list(updated_predicted_ranks.values())) + n

    stretch = initial_stretch
    total_num_proposals = num_proposals
    while len(matching) < n:
        unmatched_receivers = receivers_prefs.keys() - matching.values()
        for r in unmatched_receivers:
            updated_predicted_ranks[r] = min(updated_predicted_ranks[r] + stretch, n-1)
        stretch *= 2
        updated_predicted_matching = {r:receivers_prefs[r][match_rank] for r, match_rank in updated_predicted_ranks.items()}
        matching, num_proposals = PDA_without_doubling(proposers_prefs, receivers_prefs, updated_predicted_matching)
        total_num_proposals += num_proposals
        size += sum(list(updated_predicted_ranks.values())) + n
    
    return matching, total_num_proposals, size

def PDA_without_doubling(proposers_prefs, receivers_prefs, predicted_matching):
    pruned_proposers_prefs = prefix_prune(predicted_matching, proposers_prefs, receivers_prefs)
    matching, num_proposals = DA(pruned_proposers_prefs, receivers_prefs)
    
    return matching, num_proposals

def check_stability(matching, proposers_prefs, receivers_prefs):
    proposers, receivers = proposers_prefs.keys(), receivers_prefs.keys()
    if len(matching) < len(proposers):
        return False
    
    reverse_matching = utils.reverse_dict(matching)
    for p in proposers:
        for r in receivers:
            p_prefs, r_prefs = proposers_prefs[p], receivers_prefs[r]
            if utils.rank(p, r_prefs) < utils.rank(reverse_matching[r], r_prefs) and utils.rank(r, p_prefs) < utils.rank(matching[p], p_prefs):
                return False
    
    return True
