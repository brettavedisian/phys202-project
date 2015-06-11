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

fig_mpl_direct, ax_direct = plt.subplots(1,figsize=(5,5))
mr1=[direct_r1[k][0] for k in range(120)]
mr2=[direct_r2[k][0] for k in range(120)]
mR1=R1[0]
mR2=R2[0]
plt.sca(ax_direct)
plt.xlim(-75,75)
plt.ylim(-75,75)
plt.title('Direct Passage, S7')
plt.tick_params(axis='x', labelbottom='off')
plt.tick_params(axis='y', labelleft='off')

scatr_direct = ax_direct.scatter(mr1,mr2,c='c',s=4,label='m')
scatR_direct = ax_direct.scatter(mR1,mR2,c='r',label='S')
scatM_direct = ax_direct.scatter(0,0,c='k',label='M')

def make_frame_mpl_S7direct(t):
    newr1=[direct_r1[k][t*20] for k in range(120)]
    newr2=[direct_r2[k][t*20] for k in range(120)]
    newR1=R1[t*20]
    newR2=R2[t*20]
    # updates the data for each frame
    # this creates Nx2 matrix
    scatr_direct.set_offsets(np.transpose(np.vstack([newr1,newr2])))
    scatR_direct.set_offsets(np.transpose(np.vstack([newR1,newR2])))
    
    return mplfig_to_npimage(fig_mpl_direct)

S7direct_animation = mpy.VideoClip(make_frame_mpl_S7direct, duration=25)