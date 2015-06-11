import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timeit
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed

def direct_ic(M,gamma):
    """
    Computes the intial conditions for each star in the system in direct
    motion relative to the disrupting galaxy.
    
    Parameters
    ----------
    M, gamma: int, float
        Parameters of the initial conditions.
    Returns
    -------
    r1, r2, vr1, vr2: lists of arrays
        The initial conditions of each star in each shell.
    """
    dist = np.array([5,7.5,10,12.5,15])
    N = np.array([12,18,24,30,36])
    theta = [np.arange(0,2*np.pi,2*np.pi/i) for i in N]
    
    rr1, rr2, r1, r2, vrr1, vrr2, vr1, vr2 = [], [], [], [], [], [], [], []
    
    for i in theta:
        rr1.append(np.cos(i))
        rr2.append(np.sin(i))
        vrr1.append(np.sin(i))
        vrr2.append(-np.cos(i))

    vr = np.array([np.sqrt((gamma*M)/i) for i in dist])

    r1 = dist*rr1
    r2 = dist*rr2
    vr1 = vrr1*vr
    vr2 = vrr2*vr
    
    return r1, r2, vr1, vr2

def retro_ic(M,gamma):
    """
    Computes the intial conditions for each star in the system in retrograde
    motion relative to the disrupting galaxy.
    
    Parameters
    ----------
    M, gamma: int, float
        Parameters of the initial conditions.
    Returns
    -------
    r1, r2, vr1, vr2: lists of arrays
        The initial conditions of each star in each shell.
    """
    dist = np.array([5,7.5,10,12.5,15])
    N = np.array([12,18,24,30,36])
    theta = [np.arange(0,2*np.pi,2*np.pi/i) for i in N]
    
    rr1, rr2, r1, r2, vrr1, vrr2, vr1, vr2 = [], [], [], [], [], [], [], []
    
    for i in theta:
        rr1.append(np.cos(i))
        rr2.append(np.sin(i))
        vrr1.append(-np.sin(i))
        vrr2.append(np.cos(i))

    vr = np.array([np.sqrt((gamma*M)/i) for i in dist])

    retro_r1 = dist*rr1
    retro_r2 = dist*rr2
    retro_vr1 = vrr1*vr
    retro_vr2 = vrr2*vr
    
    return retro_r1, retro_r2, retro_vr1, retro_vr2

def parabolic_ic(M, S, gamma):
    """
    Computes the intial conditions for the disrupting galaxy.
    
    Parameters
    ----------
    M, S, gamma: int, float
        Parameters of the initial conditions.
    Returns
    -------
    R1, R2, vR1, vR2: lists of arrays
        The initial conditions of the galaxy.
    """
    R2=50
    theta = np.arctan(abs(50/R2))
    R1 = 25-(R2**2)/100
    vR = np.sqrt((2*gamma*(M+S))/np.linalg.norm([R1,R2]))
    
    if R2 == 0:
        vR1,vR2 = 0,-vR
    else:
        if R2 > 0:
            vR1 = vR*np.cos(theta)
            vR2 = -vR*np.sin(theta)
        if R2 < 0:
            vR1 = vR*np.cos(theta)
            vR2 = -vR*np.sin(theta)
            
    return R1, R2, vR1, vR2

def ics(M,S,gamma):
    """
    Compiles all the initial conditions into a single array.
    
    Parameters
    ----------
    M, S, gamma: int, float
        Parameters of the initial conditions.
    Returns
    -------
    direct_ic_total, retro_ic_total, icR, direct_star_ic, retro_star_ic: lists of arrays
        The initial conditions of the stars and galaxy combined into one array for each passage, as well as other relevent conditions.
    """
    direct_r1, direct_r2, direct_vr1, direct_vr2 = direct_ic(M,gamma)
    retro_r1, retro_r2, retro_vr1, retro_vr2 = retro_ic(M,gamma)
    R1,R2,vR1,vR2 = parabolic_ic(M,S,gamma)
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