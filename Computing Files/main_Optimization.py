import load_data
import model
import solve
import post_process
import plots


# Load all input data
data = load_data.load()

# Define variables
variables = model.define_variables(data['n_steps'])

# Build constraints
constraints = model.build_constraints(variables, data)

# Build objective
objective = model.build_objective(variables, data)

# Solve the problem
status, problem = solve.solve_problem(objective, constraints)

# Process and print results if optimal
if status == 'optimal':
    results = post_process.extract_results(variables, status)
    revenues = post_process.compute_revenues(results, data)
    post_process.print_results(revenues, results, data)

    # Generate plots
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    data['day_labels'] = [days[(data['start_weekday'] + d) % 7] for d in range(8)]  # Add day_labels to data
    plots.plot_energy_flows(results, data, revenues)
    plots.plot_financials(revenues, data)
else:
    print("No optimal solution found.")