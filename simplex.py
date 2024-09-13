import numpy as np

A = np.array(
    [
        [1, -1, 1, 0, 1, 1],
        [0, 1, -1, 1, 0, 3],
        [1, 1, -3, 7, 3, 0],
        [1, 1, 0, 0, 1, 0],
    ]
)
b = np.array([60, 17, 12, 20])
c = np.array([1, 0, 11, -10, 0, 17])
# inequality value for constraints,
inequality_array = np.array(["=", "<=", "<=", ">="])
minimise_or_maximise = "minimum"  # "minimum" or "maximum"

assert minimise_or_maximise in [
    "minimum",
    "maximum",
], f"minimise_or_maximise must be either 'minimum' or 'maximum', not {minimise_or_maximise}"
assert A.ndim == 2, f"A must be a 2D array, not {A.ndim}D"
assert b.ndim == 1, f"b must be a 1D array, not {b.ndim}D"
assert c.ndim == 1, f"c must be a 1D array, not {c.ndim}D"
assert (
    inequality_array.ndim == 1
), f"inequality_array must be a 1D array, not {inequality_array.ndim}D"
assert set(inequality_array).issubset(
    {"<=", ">=", "="}
), "inequality_array must only contain '<=', '>=' or '='"
assert (
    A.shape[0] == b.shape[0]
), f"Number of rows in A ({A.shape[0]}) must equal number of rows in b ({A.shape[0]})"
assert (
    A.shape[1] == c.shape[0]
), f"Number of columns in A ({A.shape[1]}) must equal number of rows in c ({c.shape[0]})"
assert (
    A.shape[0] == inequality_array.shape[0]
), f"Number of rows in A ({A.shape[0]}) must equal number of rows in inequality_array ({inequality_array.shape[0]})"

max_iterations = 20
M = 1e200  # artificial variable coefficient - can be set as sufficiently large as needed

# transform negative b values to positive
b_less_than_zero = b < 0

for arr in [b, A]:
    arr[b_less_than_zero] *= -1

for idx in np.where(b_less_than_zero)[0]:
    if inequality_array[idx] == "<=":
        inequality_array[idx] = ">="
    elif inequality_array[idx] == ">=":
        inequality_array[idx] = "<="

# Transform problem into standard form
if minimise_or_maximise == "maximum":
    c = -c

# Initialize basic variables
basic = np.zeros(A.shape[0], dtype=np.int64)

# Add columns for slack variables by adding a column for the number of
# constraints that are not a strict equality
slack_indices = np.where(inequality_array != "=")[0]
num_slack_variables = slack_indices.shape[0]
print(f"Adding {num_slack_variables} slack variables")
# Populate slack variables
slack_columns = np.zeros((A.shape[0], num_slack_variables))
# 1 if <= and -1 for >=
slack_columns[slack_indices, np.arange(num_slack_variables)] += np.where(
    inequality_array[slack_indices] == "<=", 1, -1
)
A = np.hstack((A, slack_columns))
c = np.hstack((c, np.zeros(num_slack_variables)))


# Find initial basic variables i.e. columns with only one non-zero value
single_nonzero_cols = np.where(np.count_nonzero(A, axis=0) == 1)[0]
row_numbers, col_numbers = np.where((A[:, single_nonzero_cols] == 1))
eligible_cols = single_nonzero_cols[col_numbers]
basic[row_numbers] = eligible_cols + 1

# create new columns for artificial variables, update basic variables
num_artificial_vars = A.shape[0] - np.sum(basic > 0)
print(f"Adding {num_artificial_vars} artificial variables")
new_columns = np.zeros((A.shape[0], num_artificial_vars))
artificial_indices = np.where(basic == 0)
new_columns[artificial_indices, np.arange(num_artificial_vars)] = 1
basic[artificial_indices] = A.shape[1] + np.arange(num_artificial_vars) + 1

A = np.hstack((A, new_columns))
c = np.hstack((c, np.full(num_artificial_vars, M)))

# Initialize nonbasic variables as all variables not in basic
nonbasic = np.setdiff1d(np.arange(A.shape[1]) + 1, basic)

# initialise x the solution vector
x = np.zeros(A.shape[1])
x[basic - 1] = b @ np.linalg.inv(A[:, basic - 1])

# Calculate objective coefficients
objective_coefficients = c - A.T @ c[basic - 1]
counter = -1
while (counter := counter + 1) < max_iterations:
    # Calculate objective coefficients
    y = np.linalg.inv(A[:, basic - 1]).T @ c[basic - 1]
    objective_coefficients[nonbasic - 1] = c[nonbasic - 1] - A[:, nonbasic - 1].T @ y
    objective_coefficients[basic - 1] = 0

    # Check for unbounded - if all nonbasic variables are <= 0
    unbounded_mask = (A[:, nonbasic - 1] <= 0).all(axis=0)
    if unbounded_mask.any():
        unbounded_columns = np.where(unbounded_mask)[0]
        for column in unbounded_columns:
            print(f"Unbounded solution space in direction of x{nonbasic[column]}")
        if (c[unbounded_columns] < 0).any():
            print("Unbounded objective function")
            break

    # If they are all positive we have an optimal solution,
    # if result still dependent on M, we have an infeasible solution
    if (objective_coefficients[nonbasic - 1] >= 0).all():
        # Optimal solution found
        if num_artificial_vars == 0 or (x[-num_artificial_vars:] == 0).all():
            print(
                f"Optimal solution found in {counter} iterations. Solution Vector: {np.round(x, 5)}"
            )
            if minimise_or_maximise == "minimum":
                print(f"Minimal value: {np.round(np.dot(c, x), 5)}")
            else:
                print(f"Maximal value: {-np.round(np.dot(c, x), 5)}")
            if (objective_coefficients[nonbasic - 1] == 0).any():
                print("There are multiple optimum values for this problem")
        else:
            print("Problem is infeasible")
        break

    # Select the entering variable as the most negative nonbasic variable
    # and solve the linear equation to get the update column
    most_negative_nonbasic = np.argmin(objective_coefficients[nonbasic - 1])
    entering = nonbasic[most_negative_nonbasic]
    update_col = np.linalg.inv(A[:, basic - 1]) @ A[:, entering - 1]
    # where a value in the update column is less than or equal to 0,
    # set the respective value to infinity divide the basic variables
    # in solution vector by the update column
    positive_vals = x[basic - 1] / np.where(update_col <= 0, np.inf, update_col)
    basic_positive_vals = np.where(positive_vals == 0, np.inf, positive_vals)
    # select the pivot as the minimum positive value
    pivot = np.min(basic_positive_vals)
    leaving = basic[basic_positive_vals == pivot][0]

    # Update basic variables
    x[basic - 1] -= pivot * update_col
    # Add the pivot to the entering column and define the leaving variable
    x[entering - 1] += pivot

    # Update basic and nonbasic variables indices
    basic[basic == leaving], nonbasic[nonbasic == entering] = entering, leaving

    cost_func_value = (
        np.round(np.dot(c, x), 5)
        if minimise_or_maximise == "minimum"
        else -np.round(np.dot(c, x), 5)
    )

    print(
        f"""
 Cost function value on iteration {counter}: {cost_func_value}
 Leaving variable:                   x_{leaving}
 Entering variable:                  x_{entering}
 """
    )

else:
    print(f"Optimal solution not found in {max_iterations} iterations")
