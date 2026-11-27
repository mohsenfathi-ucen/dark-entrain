"""Independent particle integrator and initial spatial sampling sensitivity."""
import numpy as np,pandas as pd
from scipy.integrate import solve_ivp
import CP29_SOURCE as s
s.START=-50.;s.END=50.;s.VR*=.5;tt,track,_=s.orbit_full_range()
with np.load('full_DM1/frame_+70_1_-050.0.npz') as z: U=z['U'];x=z['x'];y=z['y']
xc=(x[:-1]+x[1:])/2;yc=(y[:-1]+y[1:])/2;X,Y=np.meshgrid(xc,yc,indexing='ij');w=U[4]*np.diff(x)[:,None]*np.diff(y)[None,:];w/=w.sum();mask=w>1e-12;weights=w[mask];weights/=weights.sum();Z=np.array([X[mask],Y[mask],np.zeros(mask.sum()),-s.Q*s.OMEGA*X[mask]])
def rhs(t,z):
 X,Y,u,v=z.reshape(4,-1);dx=X-np.interp(t,tt,track[:,0]);dy=Y-np.interp(t,tt,track[:,1]);rr=np.sqrt(dx*dx+dy*dy+.25);f=-s.G2*np.interp(rr,s.rgrid,s.mgrid,left=0,right=s.mgrid[-1])/rr**3;return np.array([u,v,2*s.OMEGA*v+2*s.Q*s.OMEGA**2*X+f*dx,-2*s.OMEGA*u+f*dy]).reshape(z.shape)
def rk(z,dt):
 z=z.copy()
 for i in range(round(100/dt)):
  t=-50+i*dt;a=rhs(t,z);b=rhs(t+dt/2,z+dt*a/2);c=rhs(t+dt/2,z+dt*b/2);d=rhs(t+dt,z+dt*c);z+=dt*(a+2*b+2*c+d)/6
 return z
# Compare independent adaptive integration for a deterministic sample of 25 mass quantiles.
ids=np.unique(np.searchsorted(np.cumsum(weights),np.linspace(.02,.98,25)));z0=Z[:,ids];sol=solve_ivp(rhs,(-50,50),z0.ravel(),method='DOP853',rtol=1e-9,atol=1e-9,max_step=.1);assert sol.success;last=sol.y[:,-1].reshape(z0.shape);r=rk(z0,.05);err=np.max(np.abs(last-r),axis=1);print('Independent integrator component max errors',err);assert max(err[:2])<.02 and max(err[2:])<.002
# Two-point Gaussian quadrature in each initial cell with analytic tag weights within cell, normalized to the original cell mass.
DX,DY=np.meshgrid(np.diff(x),np.diff(y),indexing='ij');arr=[];ws=[]
for i in (-1,1):
 for j in (-1,1):
  xx=Z[0]+i*DX[mask]/(2*np.sqrt(3));yy=Z[1]+j*DY[mask]/(2*np.sqrt(3));arr.append(np.array([xx,yy,np.zeros_like(xx),-s.Q*s.OMEGA*xx]));ws.append(np.exp(-.5*((xx/3)**2+((yy-70)/35)**2)))
ws=np.array(ws);ws=ws/ws.sum(axis=0)*weights; zz=rk(np.concatenate(arr,axis=1),.05);ww=ws.ravel();xx,yy,u,v=zz;xg=(v+2*s.OMEGA*xx)/((2-s.Q)*s.OMEGA);print('quadrature',len(ww),ww@xx,ww@xg,ww[xx<-40].sum());print('cell center',pd.read_csv('CP46_PARTICLES.csv').query('dt==0.05 and dm==1 and t==50').to_dict('records'))
assert abs(ww[xx<-40].sum()-.818771)<.02 and abs(ww@xg+65.858)<2
