import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timeit
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed

from solutions import ode_solutions

def plot_static(t, whichplot, tsteps, M, S, gamma):
    """
    Plots the interactions of the galaxies at set time values.
    
    Parameters
    ----------
    t: float
        The current time t[i].
    whichplot: str
        The passage one wants to see.
    tsteps: int
        The number of steps the time will be divided into.
    M, S, gamma: int, int, float
        Parameters of the differential equation.
    Returns
    -------
    Static plot of the chosen passage.
    """
    direct_r1, direct_r2, retro_r1, retro_r2, R1, R2 = ode_solutions(t,tsteps,M,S,gamma)

    plt.figure(figsize=(10,4))
    
    o=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    j=1
    for l in o:
        mr1,mr2=[],[]
        if j==11:
            break
        if whichplot=='direct':
            for v in range(120):
                mr1.append(direct_r1[v][l])
                mr2.append(direct_r2[v][l])
        else:
            for v in range(120):
                mr1.append(retro_r1[v][l])
                mr2.append(retro_r2[v][l])
        plt.subplot(2,5,j,frame_on=False)
        plt.scatter(mr1,mr2,c='c',s=4)
        plt.scatter(R1[l],R2[l],c='r')
        plt.scatter(0,0,c='k')
        plt.xlim(-55,55)
        plt.ylim(-55,55)
        plt.tick_params(axis='x', labelbottom='off', top='off', bottom='off')
        plt.tick_params(axis='y', labelleft='off', left='off', right='off')
        j+=1

    if whichplot=='direct':
        plt.suptitle('Direct Passage', x=0.5, y=1.02, fontsize=15)
        plt.savefig("direct.png", bbox_inches='tight')
    else:
        plt.suptitle('Retrograde Passage', x=0.5, y=1.02, fontsize=15)
        plt.savefig("retrograde.png", bbox_inches='tight')
        
    plt.tight_layout()
    plt.show()