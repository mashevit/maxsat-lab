from src.sat.cnf import WCNF

def test_parse_mini_wcnf():
    inst = WCNF.parse_dimacs("data/toy/mini.wcnf")
    assert inst.n_vars == 3
    assert len(inst.clauses) == 5
    assert inst.hard_weight == 10
    hard = [cl for cl in inst.clauses if cl.is_hard]
    soft = [cl for cl in inst.clauses if not cl.is_hard]
    assert len(hard) == 2
    assert len(soft) == 3
