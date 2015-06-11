import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timeit
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed

def derivs(rvec, t, M, S, gamma = 4.4983169634398597e4):
    """
    Compute the derivatives of the position and velocity of galaxy S and
    massless points m.
    
    Parameters
    ----------
    rvec : ndarray
        The solution vector at the current time t[i]: [r[i],dr[i],R[i],dR[i]].
    t : float
        The current time t[i].
    M, S: float
        The parameters in the differential equation.
    Returns
    -------
    d2rvec : ndarray
        The vector of derviatives at t[i]: [dR[i],d2R[i],dr[i],d2r[i]].
    """
    R1 = rvec[0]
    R2 = rvec[1]
    dR1 = rvec[2]
    dR2 = rvec[3]
    r1 = rvec[4]
    r2 = rvec[5]
    dr1 = rvec[6]
    dr2 = rvec[7]
    
    r = np.sqrt(r1**2+r2**2)
    R = np.sqrt(R1**2+R2**2)
    
    d2r1 = -gamma*((M/(r**3))*r1 - (S/(np.sqrt((R1-r1)**2+(R2-r2)**2))**3)*(R1-r1) + (S/(R**3))*R1)
    d2r2 = -gamma*((M/(r**3))*r2 - (S/(np.sqrt((R1-r1)**2+(R2-r2)**2))**3)*(R2-r2) + (S/(R**3))*R2)
    d2R1 = -gamma*((M+S)/(R**3))*R1
    d2R2 = -gamma*((M+S)/(R**3))*R2
    
    d2rvec = np.array((dR1,dR2,d2R1,d2R2,dr1,dr2,d2r1,d2r2))
    
    return d2rvec