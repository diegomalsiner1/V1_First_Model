import cvxpy as cp

def solve_problem(objective, constraints):
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=True)
    return problem.status, problem