
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import  gmres, spilu, LinearOperator
import math
from typing import Tuple, List, Callable
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RectBivariateSpline

from linear_solver import solve
from heat_solver import HeatSolver

np.set_printoptions(threshold=sys.maxsize)

def main():
    
    # 1
    
    solver = HeatSolver(n_rows=320, n_cols=320)
    solver.construct_problem()
    _, res = solver.solve_problem()
    print(f"L^2 Residual: {res}")
    solver.plot()
    
    # 2 
    
    O = solver.orderConv()
    print(f"Estimated order of convergence: {O}")
    
    # 3 
    
    rs = [1.1, 1.01, 1.001, 1.0001, 1.00007, 1.00006, 1.00005, 1.00004, 1.00003, 1.00002, 1.00001]
    
    for r in rs:
        tmp_solver = HeatSolver()
        O = tmp_solver.orderConv(r)
        print(f"r = 1.0001: order = {O}")

    return 0

if __name__ == "__main__":
    main()