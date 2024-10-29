import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import  gmres, spilu, LinearOperator
import math
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RectBivariateSpline
from linear_solver import solve

class HeatSolver:
    
    L_x:        float=1,
    L_y:        float=1,
    n_rows:     int=10,
    n_cols:     int=10,
    gamma:      float=20, 
    h_f:        float=10, 
    phi_x_0:    float=10, 
    phi_x_L:    float=100,
    phi_ext:    float=300, 
    r:          float=1
    
    _A_data: np.ndarray
    _A_rows: np.ndarray
    _A_cols: np.ndarray

    _A: csc_matrix
    _b: np.ndarray
    _x: np.ndarray
    
    
    def __init__(self, 
        L_x:        float=1,
        L_y:        float=1,
        n_rows:     int=10,
        n_cols:     int=10,
        gamma:      float=20, 
        h_f:        float=10, 
        phi_x_0:    float=10, 
        phi_x_L:    float=100,
        phi_ext:    float=300, 
        r:          float=1 ):
        
        self.L_x        = L_x
        self.L_y        = L_y
        self.n_rows     = n_rows
        self.n_cols     = n_cols
        self.gamma      = gamma
        self.h_f        = h_f
        self.phi_x_0    = phi_x_0
        self.phi_x_L    = phi_x_L
        self.phi_ext    = phi_ext
        self.r          = r
        
        N = n_rows * n_cols
        num_nonzero = N + (2 * ((N - 1) - (n_rows - 1))) + (N - n_rows) + (N - n_cols)

        self._A_data = np.zeros((num_nonzero)).astype(np.float64)
        self._A_rows = np.zeros((num_nonzero), dtype=np.int64)
        self._A_cols = np.zeros((num_nonzero), dtype=np.int64)

        self._b = np.zeros((n_rows * n_cols, 1)).astype(np.float64)
        
    
    def construct_problem(self):
        
        # +--------+-------------------------------+--------+
        # |        |                               |        |
        # |    G   |               H               |    I   |
        # |        |                               |        |
        # |--------|-------------------------------|--------|
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |    D   |               E               |    F   |
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |--------|-------------------------------|--------|
        # |        |                               |        |
        # |   A    |               B               |   C    |
        # |        |                               |        |
        # +--------+-------------------------------+--------+
        

        dxdy_getter = None
        if self.r == 1:
            dxdy_getter = lambda i, j : (self.L_x / self.n_cols, self.L_y / self.n_rows)
        else:
            a_x = ((1 - self.r) / (1 - self.r**(self.n_cols / 2))) * (self.L_x / 2)
            a_y = ((1 - self.r) / (1 - self.r**(self.n_rows / 2))) * (self.L_y / 2)
            mid_i, mid_j = self.__ij_after_lower__(a_x, a_y, self.L_x/2, self.L_y/2)
            # print("Dimensions: ", self.__ij_after_upper__(a_x, a_y, self.L_x, self.L_y, mid_i, mid_j))
            dxdy_getter = lambda i, j : self.__get_dxdy__(i, j, a_x, a_y, (mid_i, mid_j))
        
        idx = [0]
        self.setACell(dxdy_getter, idx)
        for i in range(1, self.n_cols - 1):
            self.setBCell(dxdy_getter, i, idx)
        self.setCCell(dxdy_getter, idx)
        
        for j in range(1, self.n_rows - 1):
            self.setDCell(dxdy_getter, j, idx)
            for i in range(1, self.n_cols - 1):
                self.setECell(dxdy_getter, i, j, idx)
            self.setFCell(dxdy_getter, j, idx)   
            
        self.setGCell(dxdy_getter, idx)  
        for i in range(1, self.n_cols - 1):
            self.setHCell(dxdy_getter, i, idx)
        self.setICell(dxdy_getter, idx)


        A_csc = csc_matrix(
            (self._A_data, (self._A_rows, self._A_cols)), 
            shape=(self.n_rows * self.n_cols, self.n_rows * self.n_cols)
            )

        self._A = A_csc

    
    def solve_problem(self) -> Tuple[np.ndarray, float]:
        
        x, _ = solve(self._A, self._b)
        res = np.linalg.norm(self._b - self._A.dot(np.expand_dims(x, axis=1)))
        
        self._x = x.reshape(res, res)[::-1]
        
        return x, res
    

    def plot(self, num_contours: int = 20, name: str = "heatmap_plot.png"):
        _, ax = plt.subplots()

        sns.heatmap(self._x, cmap="hot", ax=ax, cbar=True)

        X, Y = np.meshgrid(np.arange(self._x.shape[1]), np.arange(self._x.shape[0]))
        ax.contour(X, Y, self._x, levels=num_contours, colors='cyan', linewidths=1)

        ax.set_xticks([0, self.n_cols], [0, self.L_x], rotation=0)
        ax.set_yticks([0, self.n_rows], [self.L_y, 0])
        
        # TOP
        plt.text(180, -10, '$-\Gamma \\frac{\partial T}{\partial y} = h_f(\phi_{\\text{ext}} - \phi)$', ha='center', va='center', fontsize=12, weight='bold') 
        
        # BOTTOM
        plt.text(180, 350, '$\\frac{\partial T}{\partial y} = 0$', ha='center', va='center', fontsize=12, weight='bold')  
        
        # LEFT 
        plt.text(-5, 180, '$T = 10$', ha='center', va='center', rotation=90, fontsize=12, weight='bold') 
        
        # RIGHT
        plt.text(330, 180, '$T = 100$', ha='center', va='center', rotation=90, fontsize=12, weight='bold')  
        
        plt.savefig(name, dpi=300, bbox_inches="tight")
        plt.show()
    
    
    def set_arrays(self, a_W, a_E, a_S, a_N, S_P, S_U, n_cols, A_data, A_rows, A_cols, b, k, idx):
    
        a_P = a_W + a_E + a_S + a_N - S_P
        
        A_data[idx[0]] = a_P
        A_rows[idx[0]] = k
        A_cols[idx[0]] = k

        b[k] = S_U
        idx[0]+=1
        
        if a_E != 0:
            A_data[idx[0]] = -a_E
            A_rows[idx[0]] = k
            A_cols[idx[0]] = k+1
            idx[0]+=1
        if a_W != 0:
            A_data[idx[0]] = -a_W
            A_rows[idx[0]] = k
            A_cols[idx[0]] = k-1
            idx[0]+=1
        if a_S != 0:
            A_data[idx[0]] = -a_S
            A_rows[idx[0]] = k
            A_cols[idx[0]] = k-n_cols
            idx[0]+=1
        if a_N != 0:
            A_data[idx[0]] = -a_N
            A_rows[idx[0]] = k
            A_cols[idx[0]] = k+n_cols
            idx[0]+=1
        
        
        
    def setACell(self, dxdy_getter, idx):
        
        dx, dy = dxdy_getter(0, 0)
        a_W = 0
        a_E = (self.gamma / dx) * dy
        a_S = 0
        a_N = (self.gamma / dy) * dx
        
        S_P = -2 * self.gamma * dy / dx
        S_u = 2 * self.gamma * self.phi_x_0 * dy / dx
        
        k = 0
        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)


    def setBCell(self, dxdy_getter, i, idx):
        
        dx, dy = dxdy_getter(i, 0)
        
        a_W = (self.gamma / dx) * dy
        a_E = (self.gamma / dx) * dy
        a_S = 0
        a_N = (self.gamma / dy) * dx
        
        S_P = 0
        S_u = 0
        
        k = i
        
        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)
        
        
    def setCCell(self, dxdy_getter, idx):
        
        dx, dy = dxdy_getter(self.n_cols-1, 0)
        
        a_W = (self.gamma / dx) * dy
        a_E = 0
        a_S = 0
        a_N = (self.gamma / dy) * dx
        
        S_P = -2 * self.gamma * dy / dx
        S_u = 2 * self.gamma * self.phi_x_L * dy / dx
        
        k = self.n_cols - 1
        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)
    
        
    def setDCell(self, dxdy_getter, j, idx):
        
        dx, dy = dxdy_getter(0, j)
        
        a_W = 0
        a_E = (self.gamma / dx) * dy
        a_S = (self.gamma / dy) * dx
        a_N = (self.gamma / dy) * dx
        
        S_P = -2 * self.gamma * dy / dx
        S_u = 2 * self.gamma * self.phi_x_0 * dy / dx
        
        k = j * self.n_cols
        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)
    
        
    def setECell(self, dxdy_getter, i, j, idx):
        
        dx, dy = dxdy_getter(i, j)
        
        a_W = (self.gamma / dx) * dy
        a_E = (self.gamma / dx) * dy
        a_S = (self.gamma / dy) * dx
        a_N = (self.gamma / dy) * dx
        
        S_P = 0
        S_u = 0
        
        k = i + j * self.n_cols
        
        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)
    
        
    def setFCell(self, dxdy_getter, j, idx):
        
        dx, dy = dxdy_getter(self.n_cols-1, j)
        a_W = (self.gamma / dx) * dy
        a_E = 0
        a_S = (self.gamma / dy) * dx
        a_N = (self.gamma / dy) * dx
        
        S_P = -2 * self.gamma * dy / dx
        S_u = 2 * self.gamma * self.phi_x_L * dy / dx
        
        k = (self.n_cols - 1) + j * self.n_cols
        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)
        
        
    def setGCell(self, dxdy_getter, idx):
        
        dx, dy = dxdy_getter(0, self.n_rows-1)
        a_W = (self.gamma / dx) * dy
        a_E = 0
        a_S = (self.gamma / dy) * dx
        a_N = 0
        
        S_P = -2 * self.gamma * dy / dx  -2 * self.gamma * dx / dy + (-self.h_f) * (dx * dy)
        S_u =  self.h_f * self.phi_ext * dx * dy + 2 * self.gamma * dy * self.phi_x_0 / dx

        
        k = (self.n_rows - 1) * self.n_cols
        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)
        
        
    def setHCell(self, dxdy_getter, i, idx):
        
        dx, dy = dxdy_getter(i, self.n_rows-1)
        a_W = (self.gamma / dx) * dy
        a_E = (self.gamma / dx) * dy
        a_S = (self.gamma / dy) * dx
        a_N = 0
        
        S_P = -2 * self.gamma * dx / dy - self.h_f * dy * dx
        S_u = 2 * self.gamma * dx * self.phi_ext / dy + self.h_f * self.phi_ext * (dx * dy )
        
        k = i + (self.n_rows - 1) * self.n_cols

        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)
        
        
    def setICell(self, dxdy_getter, idx):
        
        dx, dy = dxdy_getter(self.n_cols-1, self.n_rows-1)
        a_W = (self.gamma / dx) * dy
        a_E = 0
        a_S = (self.gamma / dy) * dx
        a_N = 0
        
        S_P = -2 * self.gamma * dy / dx  -2 * self.gamma * dx / dy + (-self.h_f) * (dx * dy)
        S_u =  (self.h_f) * self.phi_ext * (dx * dy) + 2 * self.gamma * dy * self.phi_x_L / dx
        
        k = (self.n_cols - 1) + (self.n_rows - 1) * self.n_cols
        self.set_arrays(a_W, a_E, a_S, a_N, S_P, S_u, self.n_cols, self._A_data, self._A_rows, self._A_cols, self._b, k, idx)
        
        
    def __get_displacement_x__(self, i, a_x, r, mid_i):
        
        if i < mid_i:
            return a_x * (1 - r**(i + 1)) / (1 - r)  
        else:
            lower_disp = self.__get_displacement_x__(mid_i-1, a_x, r, mid_i)
            upper_disp = self.__get_dx_lower__(mid_i-1, a_x, r) * (1 - (1/r)**((i - mid_i) + 1)) / (1 - (1/r)) 
            return lower_disp + upper_disp
        
        
    def __get_displacement_y__(self, j, a_y, r, mid_j):
        
        if j < mid_j:
            return a_y * (1 - r**(j + 1)) / (1 - r)  
        else:
            lower_disp = self.__get_displacement_x__(mid_j-1, a_y, r, mid_j)
            upper_disp = self.__get_dx_lower__(mid_j-1, a_y, r) * (1 - (1/r)**((j - mid_j) + 1)) / (1 - (1/r)) 
            return lower_disp + upper_disp
        
        
    def __get_dx_lower__(self, i, a_x, r):
        return a_x * r**i
        
        
    def __get_dy_lower__(self, j, a_y, r):
        return a_y * r**j
        
        
    def __get_dx_upper__(self, i, a_x, r, mid_i):
        return a_x * r**(mid_i - 1) / (r**(i - mid_i))
        
        
    def __get_dy_upper__(self, j, a_y, r, mid_j):
        return a_y * r**(mid_j - 1) / (r**(j - mid_j))
        
        
    def __get_dxdy__(self, i, j, a_x, a_y, r, mid):
        
        mid_i, mid_j = mid
        dx: float
        dy: float
        
        if i < mid[0]:
            dx = self.__get_dx_lower__(i, a_x, r)
        else:
            dx = self.__get_dx_upper__(i, a_x, r, mid_i)

        if j < mid[1]:
            dy = self.__get_dy_lower__(j, a_y, r)
        else:
            dy = self.__get_dy_upper__(j, a_y, r, mid_j)
            
        return dx, dy
            
            
    def __i_after_lower__(self, a_x, r, x_f):
        prefix_sum_i = 0
        i = 0
        while True:
            prefix_sum_i += self.__get_dx_lower__(i, a_x, r)
            if prefix_sum_i > x_f:
                return i
            i += 1
            
            
    def __j_after_lower__(self, a_y, r, y_f):
        prefix_sum_j = 0
        j = 0
        while True:
            prefix_sum_j += self.__get_dy_lower__(j, a_y, r)
            if prefix_sum_j > y_f:
                return j
            j += 1
            
            
    def __i_after_upper__(self, a_x, r, x_f, mid_i):
        i = 0
        while True:
            if self.__get_displacement_x__(i, a_x, r, mid_i) > x_f:
                return i
            i += 1
            
    def __j_after_upper__(self, a_y, r, y_f, mid_j):
        j = 0
        while True:
            if self.__get_displacement_x__(j, a_y, r, mid_j) > y_f:
                return j
            j += 1
        
        
    def __ij_after_lower__(self, a_x, a_y, r, x_f, y_f):
        return self.__i_after_lower__(a_x, r, x_f), self.__j_after_lower__(a_y, r, y_f)


    def __ij_after_upper__(self, a_x, a_y, r, x_f, y_f, mid_i, mid_j):
        return self.__i_after_upper__(a_x, r, x_f, mid_i), self.__j_after_upper__(a_y, r, y_f, mid_j)
    
    
    def orderConv(r: float = 1, exact_res: float = 320, coarse_res: float = 80, fine_res: float = 160) -> float:
        exact_res: int = 320
        coarse_res: int = 80
        fine_res: int = 160
        
        exact_solver = HeatSolver(n_rows=exact_res, n_cols=exact_res, r=r)
        coarse_solver = HeatSolver(n_rows=coarse_res, n_cols=coarse_res, r=r)
        fine_solver = HeatSolver(n_rows=fine_res, n_cols=fine_res, r=r)
        
        exact_solver.construct_problem()
        coarse_solver.construct_problem()
        fine_solver.construct_problem()
        
        x_coarse, _ = coarse_solver.solve_problem()
        x_fine, _ = fine_solver.solve_problem()
        x_exact, _ = exact_solver.solve_problem()

        linsp_coarse = np.linspace(0, 1, coarse_res)
        linsp_fine = np.linspace(0, 1, fine_res)
        linsp_exact = np.linspace(0, 1, exact_res)

        x_coarse_interpolator = RectBivariateSpline(linsp_coarse, linsp_coarse, x_coarse.reshape(coarse_res, coarse_res))

        x_coarse_interp = x_coarse_interpolator(linsp_exact, linsp_exact)

        x_fine_interpolator = RectBivariateSpline(linsp_fine, linsp_fine, x_fine.reshape(fine_res, fine_res))

        x_fine_interp = x_fine_interpolator(linsp_exact, linsp_exact)

        err_coarse = np.linalg.norm(x_exact - x_coarse_interp.flatten()) / coarse_res
        err_fine = np.linalg.norm(x_exact - x_fine_interp.flatten()) / fine_res

        O = (np.log(np.abs(err_coarse / err_fine))) / (np.log((1 / coarse_res) / (1 / fine_res)))
        
        return O

    