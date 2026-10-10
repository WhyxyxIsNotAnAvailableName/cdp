def select_semantic_factors(factors, budget):
    return sorted(factors, key=lambda factor: factor["factor_id"])[:budget - 1]
