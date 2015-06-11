import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timeit
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

from S5_icsandsolutions import *

gamma = 4.4983169634398597e4
tsteps = 1000
t = np.linspace(0,1.5,tsteps)
M = 10
S = 10

S5direct_r1, S5direct_r2, S5retro_r1, S5retro_r2, S5R1, S5R2, S5vR1, S5vR2 = S5_ode_solutions(t,tsteps,M,S,gamma)

fig_mpl_S5direct, ax_S5direct = plt.subplots(1,figsize=(5,5))
mr1=[S5direct_r1[k][0] for k in range(120)]
mr2=[S5direct_r2[k][0] for k in range(120)]
mR1=S5R1[0]
mR2=S5R2[0]
plt.sca(ax_S5direct)
plt.xlim(-75,75)
plt.ylim(-75,75)
plt.title('Direct Passage, S5')
plt.tick_params(axis='x', labelbottom='off')
plt.tick_params(axis='y', labelleft='off')

scatr_S5direct = ax_S5direct.scatter(mr1,mr2,c='c',s=4,label='m')
scatR_S5direct = ax_S5direct.scatter(mR1,mR2,c='r',label='S')
scatM_S5direct = ax_S5direct.scatter(0,0,c='k',label='M')

def make_frame_mpl_S5direct(t):
    newr1=[S5direct_r1[k][t*20] for k in range(120)]
    newr2=[S5direct_r2[k][t*20] for k in range(120)]
    newR1=S5R1[t*20]
    newR2=S5R2[t*20]
    # updates the data for each frame
    # this creates Nx2 matrix
    scatr_S5direct.set_offsets(np.transpose(np.vstack([newr1,newr2])))
    scatR_S5direct.set_offsets(np.transpose(np.vstack([newR1,newR2])))
    
    return mplfig_to_npimage(fig_mpl_S5direct)

S5direct_animation = mpy.VideoClip(make_frame_mpl_S5direct, duration=25)