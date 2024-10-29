
import numpy as np
import sys
from heat_solver import HeatSolver
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

def main():
    
    # 1
    print("\nQuestion 1")
    
    solver = HeatSolver(n_rows=320, n_cols=320)
    solver.construct_problem()
    _, res = solver.solve_problem()
    print(f"L^2 Residual: {res}")
    solver.plot()
    
    # 2 
    print("\nQuestion 2")
    O = solver.orderConv()
    print(f"Estimated order of convergence: {O}")
    
    # 3 
    print("\nQuestion 3")
    x = 1 + np.logspace(np.log10(0.1), np.log10(0.00001), num=10, base=10)
    y = []
    for r in x:
        s = HeatSolver()
        O = s.orderConv(r)
        print(f"r: {r}, order of convergence: {O}")
        y.append(O)

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linestyle='-', color='b')

    ax.set_xlabel("Inflation r")
    ax.set_ylabel("Order of convergence")
    
    plt.show()
    
    fig.savefig("order_of_convergence.png")

    return 0

if __name__ == "__main__":
    main()