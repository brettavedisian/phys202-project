import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timeit
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed

from solutions import ode_solutions

gamma = 4.4983169634398597e4
tsteps = 1000
t = np.linspace(0,1.5,tsteps)
M = 10
S = 10

direct_r1, direct_r2, retro_r1, retro_r2, R1, R2 = ode_solutions(t,tsteps,M,S,gamma)

def plot_ode(direct_r1, direct_r2, retro_r1, retro_r2, R1, R2, whichplot, n=0):
    """
    Plots the stars and galaxies positions as a function of time, controlled by the user.
    
    Parameters
    ----------
    direct_r1, direct_r2, retro_r1, retro_r2, R1, R2: lists of arrays
        Each list contains an array for the solution of each star at each time. Also, the solutions for the galaxy.
    whichplot: str
        The user chooses which passage to see.
    n: int
        Time values chosen by the user.
    Returns
    -------
    Plot to be used with interactive.
    """    
    plt.figure(figsize=(9,9))
    plt.scatter(0, 0, label='M', c='k')
    
    if whichplot=='direct':
        for o in range(120):
            plt.scatter(direct_r1[o][n],direct_r2[o][n], label='m', c='c', s=5)
        plt.title('Direct Passage')
    else:
        for o in range(120):
            plt.scatter(retro_r1[o][n],retro_r2[o][n], label='m', c='c', s=5)
        plt.title('Retrograde Passage')
        
    plt.scatter(R1[n], R2[n], label='S', c='r100')
    plt.tick_params(axis='x', labelbottom='off')
    plt.tick_params(axis='y', labelleft='off')
    
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    
    plt.show()