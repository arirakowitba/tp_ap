import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Let's inspect the exact sub matrix and test all possible integer labelings (-1, 0, 1)
# 1 = hit (+1), -1 = miss (-1), 0 = private test set (0)
# Since public is 70% and private is 30%, each item has probability 0.70 of being in public.
from solve_todays_sub_matrix import A, b, all_ids, id_to_idx

print("A shape:", A.shape)
print("b:", b)

# For each item, public effect can only be +1 (true condition is flipped), -1 (true condition is anchor), or 0 (in private set)
# Let's see if we can find all ternary vectors x in {-1, 0, 1}^35 such that A @ x = b!
print("\nSolving for ternary vectors x in {-1, 0, 1}^35...")

# Since A has rank 5 and 35 variables, let's use z3 or a simple constraint satisfaction or integer programming!
try:
    from z3 import Solver, Int, Or, And, sat
    s = Solver()
    vars = [Int(f"x_{i}") for i in range(len(all_ids))]
    for v in vars:
        s.add(Or(v == -1, v == 0, v == 1))
    for row in range(A.shape[0]):
        expr = sum(int(A[row, col]) * vars[col] for col in range(len(all_ids)))
        s.add(expr == int(b[row]))
    
    # Check satisfiability
    count = 0
    solutions = []
    while s.check() == sat and count < 20:
        m = s.model()
        sol = [m.evaluate(vars[i]).as_long() for i in range(len(all_ids))]
        solutions.append(sol)
        # Block this solution
        s.add(Or([vars[i] != sol[i] for i in range(len(all_ids))]))
        count += 1
    print(f"Found {count} ternary solutions in Z3!")
    if solutions:
        df_sol = pd.DataFrame(solutions, columns=all_ids)
        print("Mean estimated effect across first 20 solutions:")
        print(df_sol.mean().sort_values(ascending=False))
except Exception as e:
    print("Z3 not available or error:", e)
