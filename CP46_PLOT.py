from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import CP29_SOURCE as s
from matplotlib.patches import Circle
r=Path(__file__).resolve().parent;m=r/'manuscript';D=pd.read_csv(r/'CP46_FIELDS.csv');P=pd.read_csv(r/'CP46_PARTICLES.csv');p=P[(P.dt==.05)&(P.dm==1)];g=D[D.dm==1];c=D[D.dm==0]
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(9,6.8),layout='constrained')
a=ax[0,0];a.plot(g.t,g.x,'o-',ms=3,label='Gas centroid');a.fill_between(g.t.to_numpy(),g.x_q10.to_numpy(),g.x_q90.to_numpy(),alpha=.15);a.plot(p.t,p.x,'s--',ms=3,label='Pressureless centroid');a.axhline(-40,color='.5',ls=':');a.set_ylabel('Radial position (pc)');a.legend(fontsize=9,loc='lower left',framealpha=.85)
a=ax[0,1];a.plot(g.t,g.xg,'o-',ms=3,label='Gas');a.fill_between(g.t.to_numpy(),g.xg_q10.to_numpy(),g.xg_q90.to_numpy(),alpha=.15);a.plot(p.t,p.xg,'s--',ms=3,label='Pressureless');a.axhline(0,color='.5',ls=':');a.set_ylabel('Guiding-centre proxy (pc)');a.legend(fontsize=9,loc='lower right',framealpha=.85)
a=ax[1,0];a.plot(g.t,100*(g.in_40.to_numpy()-c.in_40.to_numpy()),'o-',ms=3,label='Gas, excess inventory');a.plot(p.t,100*p.in_40,'s--',ms=3,label='Pressureless');a.plot(g.t,100*g.inner_far_rj,'^-',ms=3,label='Gas, outside Hill radius');a.plot(g.t,100*g.joint_rj,'d-',ms=3,label='Also positive proxy');a.set_ylabel('Initial tagged mass (%)');a.legend(fontsize=8.5,loc='upper left',framealpha=.85)
a=ax[1,1];
for key,label in [('free','Epicyclic'),('subhalo','Remnant'),('pressure','Pressure estimate'),('total','Sum')]:a.plot(g.t,g[key],'.-',ms=3,label=label)
a.axhline(0,color='.5',ls=':');a.set_ylabel(r'Radial acceleration (pc Myr$^{-2}$)');a.legend(fontsize=8.5,loc='lower right',framealpha=.85)
for a in ax.flat:a.set_xlabel('Time (Myr)');a.axvline(0,color='.6',ls=':',lw=.7);a.grid(alpha=.15)
fig.savefig(m/'gas_particle_comparison.pdf');plt.close(fig)
# Regenerate slow thermal maps and joint mass distributions from independently evolved wide mesh.
def fields(dm,t):
 with np.load(r/f'full_DM{dm}/frame_+70_{dm}_{t:+06.1f}.npz') as z:U=z['U'];x=z['x'];y=z['y'];mx=float(z['dm_x']);my=float(z['dm_y'])
 rho=U[0];p=(s.GAMMA-1)*(U[3]-.5*(U[1]**2+U[2]**2)/rho);T=s.Tw*p/(s.P0_RAD*rho);W=U[4]*np.diff(x)[:,None]*np.diff(y)[None,:];return U,x,y,T,W,mx,my
fig,ax=plt.subplots(2,2,figsize=(8.4,7.1),layout='constrained');fields4=[fields(1,t) for t in (0,20,30,50)];lo=min(v[3].min() for v in fields4);hi=max(v[3].max() for v in fields4)
for a,t,(U,x,y,T,W,mx,my) in zip(ax.flat,(0,20,30,50),fields4):
 im=a.pcolormesh(x,y,T.T,norm=LogNorm(lo,hi),cmap='inferno',shading='flat',rasterized=True);a.contour((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2,U[4].T,levels=[.1,1,10],colors=['#9cecfb','#32c9df','#00a6cf'],linewidths=.8);a.add_patch(Circle((mx,my),40,fill=False,edgecolor='white',lw=.9));a.plot(mx,my,'w+',ms=7);a.axvline(-40,c='white',ls='--',lw=.8);a.set(xlabel='x (pc)',ylabel='y (pc)',title=f't = {t} Myr',xlim=(-500,500),ylim=(-500,500),aspect='equal')
fig.colorbar(im,ax=ax.ravel().tolist(),label='Temperature (K)',shrink=.8);fig.savefig(m/'reconstructed_slow_maps.pdf');plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(8.5,6.5),layout='constrained');binsx=np.arange(-500,505,5);binsT=np.geomspace(10,3e4,121)
for a,(dm,t) in zip(ax.flat,[(0,0),(1,0),(0,50),(1,50)]):
 U,x,y,T,W,_,_=fields(dm,t);X=np.broadcast_to(((x[:-1]+x[1:])/2)[:,None],W.shape);H,*_=np.histogram2d(X.ravel(),T.ravel(),bins=(binsx,binsT),weights=W.ravel()/W.sum());assert H.sum()>.9999
 im=a.pcolormesh(binsx,binsT,H.T,norm=LogNorm(1e-5,.1),cmap='viridis',shading='flat',rasterized=True);a.axvline(-40,c='white',ls='--',lw=.8);a.set(yscale='log',xlim=(-350,150),xlabel='x (pc)',ylabel='Temperature (K)',title=f'{"Encounter" if dm else "Control"}, {t} Myr')
fig.colorbar(im,ax=ax.ravel().tolist(),label='Tagged mass fraction per bin',shrink=.8);fig.savefig(m/'slow_joint_density.pdf');plt.close(fig)
