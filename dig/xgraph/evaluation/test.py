from sympy import im


def test_array_best():
    from dig.xgraph.evaluation.metrics_tg import array_best
    input = [2, 3, 4, 1, 5]
    output = array_best(input)
    assert [2, 3, 4, 4, 5] == output