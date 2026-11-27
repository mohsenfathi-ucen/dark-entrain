"""Finite-window gravity-only comparison and saved-field diagnostics. No new hydro evolution."""
from pathlib import Path
import numpy as np, pandas as pd, json
import CP29_SOURCE as s
root=Path(__file__).resolve().parent
s.START=-50.;s.END=50.;s.VR*=.5
tt,track,_=s.orbit_full_range(); k2=2*(2-s.Q)*s.OMEGA**2
rj=(s.G2*s.mgrid[-1]/(2*s.Q*s.OMEGA**2))**(1/3)
def force(t,X,Y):
 mx,my=[np.interp(t,tt,track[:,i]) for i in (0,1)]; dx=X-mx;dy=Y-my;rr=np.sqrt(dx*dx+dy*dy+.25);fac=-s.G2*np.interp(rr,s.rgrid,s.mgrid,left=0,right=s.mgrid[-1])/rr**3
 return fac*dx,fac*dy,np.hypot(dx,dy)
def quant(v,w,qs=(.1,.5,.9)):
 a=np.argsort(v.ravel());return np.interp(qs,np.cumsum(w.ravel()[a])/w.sum(),v.ravel()[a])
rows=[]
for dm in (0,1):
 for p in sorted((root/f'full_DM{dm}').glob('frame_*.npz')):
  with np.load(p) as z: U=z['U'];x=z['x'];y=z['y'];t=float(z['t'])
  xc=(x[1:]+x[:-1])/2;yc=(y[1:]+y[:-1])/2;X,Y=np.meshgrid(xc,yc,indexing='ij');rho=U[0];vx=U[1]/rho;vy=U[2]/rho;P=(s.GAMMA-1)*(U[3]-.5*rho*(vx*vx+vy*vy)); W=U[4]*np.diff(x)[:,None]*np.diff(y)[None,:];W/=W.sum();xg=(vy+2*s.OMEGA*X)/((2-s.Q)*s.OMEGA);ax,ay,R=force(t,X,Y);free=-k2*(X-xg);pressure=-np.gradient(P,xc,axis=0,edge_order=2)/rho; vrel=np.hypot(vx-np.interp(t,tt,np.gradient(track,tt,axis=0)[:,0]),vy-np.interp(t,tt,np.gradient(track,tt,axis=0)[:,1])); mx,my=[np.interp(t,tt,track[:,i]) for i in (0,1)];ux,uy=[np.interp(t,tt,np.gradient(track,tt,axis=0)[:,i]) for i in (0,1)];vrel_inertial=np.hypot(vx-ux-s.OMEGA*(Y-my),vy-uy+s.OMEGA*(X-mx)); T=s.Tw*P/(s.P0_RAD*rho)
  avg=lambda a:float(np.sum(W*a)); f=lambda a:float(W[a].sum());d=dict(dm=dm,t=t,x=avg(X),xg=avg(xg),vx=avg(vx),free=avg(free),subhalo=dm*avg(ax),pressure=avg(pressure),Tmean=avg(T));d['total']=d['free']+d['subhalo']+d['pressure']
  for a in (20,40,80,120,150): d[f'in_{a}']=f(X<-a)
  for a,n in [(40,'40'),(80,'80'),(120,'120'),(s.rt,'rt'),(rj,'rj')]:d['near_'+n]=f(R<a);d['inner_far_'+n]=f((X<-40)&(R>=a));d['joint_'+n]=f((X<-40)&(R>=a)&(xg>0));d['joint40_'+n]=f((X<-40)&(R>=a)&(xg>40))
  for name,field in [('x',X),('xg',xg),('vrel_rot',vrel),('vrel_inertial',vrel_inertial),('T',T),('concentration',U[4]/rho)]:
   for q,v in zip((10,50,90),quant(field,W)):d[f'{name}_q{q}']=v
  mask=U[4]/rho>.5;d['high_c_mass']=f(mask);d['high_c_Tmean']=float(np.sum(W[mask]*T[mask])/W[mask].sum()) if mask.any() else float('nan');rows.append(d)
D=pd.DataFrame(rows).sort_values(['dm','t']); D.to_csv(root/'CP46_FIELDS.csv',index=False)
# Weighted particles at cell centers; initial velocities exactly from the hydrodynamic initial state.
with np.load(root/'full_DM1/frame_+70_1_-050.0.npz') as z:U=z['U'];x=z['x'];y=z['y']
X,Y=np.meshgrid((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2,indexing='ij');w=U[4]*np.diff(x)[:,None]*np.diff(y)[None,:];w/=w.sum();mask=w>1e-12;discard=1-w[mask].sum();w=w[mask];w/=w.sum();init=np.array([X[mask],Y[mask],(U[1]/U[0])[mask],(U[2]/U[0])[mask]])
def rhs(t,z,dm):
 X,Y,u,v=z;ax,ay,_=force(t,X,Y);return np.array([u,v,2*s.OMEGA*v+2*s.Q*s.OMEGA**2*X+dm*ax,-2*s.OMEGA*u+dm*ay])
P=[]
for dt in (.1,.05):
 for dm in (0,1):
  z=init.copy();n=round(100/dt)
  for i in range(n+1):
   t=-50+i*dt
   if i%round(5/dt)==0:
    X,Y,u,v=z;xg=(v+2*s.OMEGA*X)/((2-s.Q)*s.OMEGA);_,_,R=force(t,X,Y);p=dict(dt=dt,dm=dm,t=t,x=float(w@X),xg=float(w@xg),vx=float(w@u),in_40=float(w[X<-40].sum()),inner_far_rt=float(w[(X<-40)&(R>=s.rt)].sum()),inner_far_rj=float(w[(X<-40)&(R>=rj)].sum()),joint_rt=float(w[(X<-40)&(R>=s.rt)&(xg>0)].sum()));P.append(p)
   if i==n:break
   a=rhs(t,z,dm);b=rhs(t+dt/2,z+dt*a/2,dm);c=rhs(t+dt/2,z+dt*b/2,dm);d=rhs(t+dt,z+dt*c,dm);z+=dt*(a+2*b+2*c+d)/6
P=pd.DataFrame(P);P.to_csv(root/'CP46_PARTICLES.csv',index=False)
print('rj',rj,'particles',len(w),'discarded',discard)
print(D[(D.dm==1)&(D.t>=0)][['t','x','xg','vx','free','subhalo','pressure','in_40','inner_far_rt','inner_far_rj','joint40_rj','T_q50','T_q90']].to_string(index=False))
print(P[(P.t==50)].to_string(index=False))
print('max particle dt diff',P[P.dt==.1].reset_index(drop=True).in_40.sub(P[P.dt==.05].reset_index(drop=True).in_40).abs().max())
