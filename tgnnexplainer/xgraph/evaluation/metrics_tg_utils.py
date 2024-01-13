


def fidelity_inv_tg(ori_probs, important_probs):
    """
    important_probs: prediction with only searched important events
    Generally the smaller the better.
    """
    # if ori_probs >= 0.5: # NOTE: not 0.5, should be 0!
    if ori_probs >= 0: # logit
        res = important_probs - ori_probs
    else: res = ori_probs - important_probs

    return res

def sparsity_tg(tree_node, candidate_number: int):
    # return 1.0 - len(tree_node.coalition) / candidate_number
    return len(tree_node.coalition) / candidate_number