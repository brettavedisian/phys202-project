import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timeit
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

from S7_icsandsolutions import *

gamma = 4.4983169634398597e4
tsteps = 1000
t = np.linspace(0,1.5,tsteps)
M = 10
S = 10

direct_r1, direct_r2, retro_r1, retro_r2, R1, R2, vR1, vR2 = S7_ode_solutions(t,tsteps,M,S,gamma)

fig_mpl_retro, ax_retro = plt.subplots(1,figsize=(5,5))
mr1=[retro_r1[k][0] for k in range(120)]
mr2=[retro_r2[k][0] for k in range(120)]
mR1=R1[0]
mR2=R2[0]
plt.sca(ax_retro)
plt.xlim(-75,75)
plt.ylim(-75,75)
plt.title('Retrograde Passage')
plt.tick_params(axis='x', labelbottom='off')
plt.tick_params(axis='y', labelleft='off')

scatr_retro = ax_retro.scatter(mr1,mr2,c='c',s=4,label='m')
scatR_retro = ax_retro.scatter(mR1,mR2,c='r',label='S')
scatM_retro = ax_retro.scatter(0,0,c='k',label='M')

def make_frame_mpl_S7retro(t):
    newr1=[retro_r1[k][t*20] for k in range(120)]
    newr2=[retro_r2[k][t*20] for k in range(120)]
    newR1=R1[t*20]
    newR2=R2[t*20]
    # updates the data for each frame
    # this creates Nx2 matrix
    scatr_retro.set_offsets(np.transpose(np.vstack([newr1,newr2])))
    scatR_retro.set_offsets(np.transpose(np.vstack([newR1,newR2])))
    
    return mplfig_to_npimage(fig_mpl_retro)

S7retro_animation = mpy.VideoClip(make_frame_mpl_S7retro, duration=25)