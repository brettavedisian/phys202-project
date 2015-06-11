import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timeit
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed

from initialconditions import *
from derivsfunc import derivs

def ode_solutions(t,tsteps,M,S,gamma):
    """
    Solve the differentials with an array of initial conditions and returns lists of positions and velocities.
    
    Parameters
    ----------
    t: float
        The current time t[i].
    tsteps: int
        The number of times between [0,t] the solution is calculated.
    M, S, gamma: float
        Parameters of the differential equation.
    Returns
    -------
    direct_r1, direct_r2, retro_r1, retro_r2, R1, R2: lists of arrays
        Lists of 120 arrays each with tsteps number of solutions; one array for each star.
    """
    direct_ic_total, retro_ic_total, icR, direct_star_ic, retro_star_ic = ics(M,S,gamma)
    direct_r1,direct_r2,direct_vr1,direct_vr2 = [], [], [], []
    direct_complete_sol = []
    retro_r1,retro_r2,retro_vr1,retro_vr2 = [], [], [], []
    retro_complete_sol = []
    
    for i in range(120):
        direct_ic = np.append(icR,direct_star_ic[i])
        direct_solution = odeint(derivs, direct_ic, t, args=(M, S), atol=1e-5, rtol=1e-5)
        direct_complete_sol.append(direct_solution)
        direct_r1.append(direct_complete_sol[i][0:tsteps,4])
        direct_r2.append(direct_complete_sol[i][0:tsteps,5])
        direct_vr1.append(direct_complete_sol[i][0:tsteps,6])
        direct_vr2.append(direct_complete_sol[i][0:tsteps,7])
        
        retro_ic = np.append(icR,retro_star_ic[i])
        retro_solution = odeint(derivs, retro_ic, t, args=(M, S), atol=1e-5, rtol=1e-5)
        retro_complete_sol.append(retro_solution)
        retro_r1.append(retro_complete_sol[i][0:tsteps,4])
        retro_r2.append(retro_complete_sol[i][0:tsteps,5])
        retro_vr1.append(retro_complete_sol[i][0:tsteps,6])
        retro_vr2.append(retro_complete_sol[i][0:tsteps,7])
        
    R1 = direct_complete_sol[0][0:tsteps,0]
    R2 = direct_complete_sol[0][0:tsteps,1]
    vR1 = direct_complete_sol[0][0:tsteps,2]
    vR2 = direct_complete_sol[0][0:tsteps,3]
    
    return direct_r1, direct_r2, retro_r1, retro_r2, R1, R2