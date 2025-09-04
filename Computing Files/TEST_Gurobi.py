import sys


def check_gurobi():
	print("python:", sys.executable)
	try:
		import gurobipy as gp
		print("gurobi_version:", gp.gurobi.version())
		# Trivial model to exercise the license
		m = gp.Model()
		x = m.addVar(name="x")
		m.setObjective(x, gp.GRB.MAXIMIZE)
		m.addConstr(x <= 1)
		m.optimize()
		status = getattr(m, "Status", getattr(m, "status", None))
		print("gurobi_status:", status)
		print("license_ok:", status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT))
	except Exception as e:
		print("gurobi_error:", repr(e))


def check_pypsa_with_gurobi():
	try:
		import pypsa
		import pandas as pd
		n = pypsa.Network()
		n.set_snapshots(pd.RangeIndex(1))
		n.add("Bus", "b")
		n.add("Generator", "g", bus="b", p_nom=1, marginal_cost=1)
		n.add("Load", "l", bus="b", p_set=0.5)
		n.optimize.create_model()
		n.optimize.solve_model(solver_name="gurobi")
		print("pypsa_solved_with_gurobi:", n.is_solved)
	except Exception as e:
		print("pypsa_gurobi_error:", repr(e))


if __name__ == "main" or __name__ == "__main__":
	check_gurobi()
	check_pypsa_with_gurobi()

