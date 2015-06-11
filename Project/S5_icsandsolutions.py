import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timeit
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed

from derivsfunc import *
from initialconditions import direct_ic, retro_ic

def S5_parabolic_ic(M, S, gamma):
    """
    Computes the S5 intial conditions for the disrupting galaxy.
    """
    R2 = -55
    R1 = 25-(R2**2)/100
    vR = np.sqrt((2*gamma*(M+S))/np.linalg.norm([R1,R2]))
    
    if R2 == 0:
        vR1,vR2 = 0,vR
    else:
        theta = np.arctan((50/R2))
        if R2 > 0:
            vR1 = vR*np.cos(theta)
            vR2 = -vR*np.sin(theta)
        if R2 < 0:
            vR1 = vR*np.cos(theta)
            vR2 = -vR*np.sin(theta)

    return R1, R2, vR1, vR2

def S5_ics(M,S,gamma):
    """
    Compiles all the S5 initial conditions into a single array.
    """
    direct_r1, direct_r2, direct_vr1, direct_vr2 = direct_ic(M,gamma)
    retro_r1, retro_r2, retro_vr1, retro_vr2 = retro_ic(M,gamma)
    R1,R2,vR1,vR2 = S5_parabolic_ic(M,S,gamma)
    icR = np.array([R1,R2,vR1,vR2])
    
    direct_mr1 = np.hstack((direct_r1[0],direct_r1[1],direct_r1[2],direct_r1[3],direct_r1[4]))
    direct_mr2 = np.hstack((direct_r2[0],direct_r2[1],direct_r2[2],direct_r2[3],direct_r2[4]))
    direct_mvr1 = np.hstack((direct_vr1[0],direct_vr1[1],direct_vr1[2],direct_vr1[3],direct_vr1[4]))
    direct_mvr2 = np.hstack((direct_vr2[0],direct_vr2[1],direct_vr2[2],direct_vr2[3],direct_vr2[4]))

    retro_mr1 = np.hstack((retro_r1[0],retro_r1[1],retro_r1[2],retro_r1[3],retro_r1[4]))
    retro_mr2 = np.hstack((retro_r2[0],retro_r2[1],retro_r2[2],retro_r2[3],retro_r2[4]))
    retro_mvr1 = np.hstack((retro_vr1[0],retro_vr1[1],retro_vr1[2],retro_vr1[3],retro_vr1[4]))
    retro_mvr2 = np.hstack((retro_vr2[0],retro_vr2[1],retro_vr2[2],retro_vr2[3],retro_vr2[4]))
    
    direct_star_ic = np.transpose(np.vstack((direct_mr1,direct_mr2,direct_mvr1,direct_mvr2)))
    retro_star_ic = np.transpose(np.vstack((retro_mr1,retro_mr2,retro_mvr1,retro_mvr2)))
    direct_ic_total = np.append(icR,direct_star_ic)
    retro_ic_total = np.append(icR,retro_star_ic)
    
    return direct_ic_total, retro_ic_total, icR, direct_star_ic, retro_star_ic

def S5_ode_solutions(t,tsteps,M,S,gamma):
    """
    Solve the differentials with an array of initial conditions of the S5 case and returns lists of positions and velocities.
    
    Parameters
    ----------
    t: float
        The current time t[i].
    tsteps: int
        The number of times between [0,t] the solution is calculated.
    M, S: float
        Parameters of the differential equation.
        
    Returns
    -------
    direct_r1, direct_r2, retro_r1, retro_r2, R1, R2: lists of arrays
        Lists of 120 arrays each with tsteps number of solutions; one array for each star.
    """
    direct_ic_total, retro_ic_total, icR, direct_star_ic, retro_star_ic = S5_ics(M,S,gamma)
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
    
    return direct_r1, direct_r2, retro_r1, retro_r2, R1, R2, vR1, vR2