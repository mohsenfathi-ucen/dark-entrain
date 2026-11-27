# DARK-ENTRAIN CP15 | standalone one-cell Euler/HLLC hydrodynamic onset, CP14 physical model

from pathlib import Path

import numpy as np

import math, json, csv, zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

GAMMA=5./3.

ROOT=Path.cwd()

def primitives(U):
    rho=np.maximum(U[0],1e-9)
    vx=U[1]/rho;vy=U[2]/rho
    ek=.5*(U[1]**2+U[2]**2)/rho
    p=np.maximum((GAMMA-1)*(U[3]-ek),1e-8)
    a=np.sqrt(GAMMA*p/rho)
    return rho,vx,vy,p,a

def flux(U,axis):
    rho,vx,vy,p,a=primitives(U)
    if axis=='x':
        F=np.array([U[1],U[1]*vx+p,U[2]*vx,(U[3]+p)*vx,U[4]*vx]); sp=np.abs(vx)+a
    else:
        F=np.array([U[2],U[1]*vy,U[2]*vy+p,(U[3]+p)*vy,U[4]*vy]); sp=np.abs(vy)+a
    return F,sp

def _minmod(a,b):
    s=np.sign(a)+np.sign(b)
    return np.where(np.abs(s)==2,np.sign(a)*np.minimum(np.abs(a),np.abs(b)),0.)

def slope_limited(U, axis):
    # minmod + no extrapolation at the outermost cells
    d0=U-np.roll(U,1,axis=axis)
    d1=np.roll(U,-1,axis=axis)-U
    s=_minmod(d0,d1)
    if axis==1: s[:,0,:]=0.;s[:,-1,:]=0.
    else: s[:,:,0]=0.;s[:,:,-1]=0.
    return s

def pressure_and_sound(U):
 rho=U[0];vn=U[1]/rho;vt=U[2]/rho
 p=(GAMMA-1)*(U[3]-.5*(U[1]**2+U[2]**2)/rho)
 return rho, vn,vt,p,np.sqrt(GAMMA*p/rho)

def hllc_flux(L,R,axis):
 # Normal velocities and tangential moments are reordered for x/y; passive tag advected with mass flux.
 if axis=='y':
  L=L[[0,2,1,3,4]];R=R[[0,2,1,3,4]]
 rhoL,unL,utL,pL,cL=pressure_and_sound(L)
 rhoR,unR,utR,pR,cR=pressure_and_sound(R)
 if np.any(pL<=0) or np.any(pR<=0) or np.any(rhoL<=0) or np.any(rhoR<=0):
  raise RuntimeError('nonphysical reconstructed interface')
 fL=np.array([L[1],L[1]*unL+pL,L[2]*unL,(L[3]+pL)*unL,L[4]*unL])
 fR=np.array([R[1],R[1]*unR+pR,R[2]*unR,(R[3]+pR)*unR,R[4]*unR])
 sL=np.minimum(unL-cL,unR-cR)
 sR=np.maximum(unL+cL,unR+cR)
 denom=rhoL*(sL-unL)-rhoR*(sR-unR)
 sM=(pR-pL+rhoL*unL*(sL-unL)-rhoR*unR*(sR-unR))/denom
 pStarL=pL+rhoL*(sL-unL)*(sM-unL)
 pStarR=pR+rhoR*(sR-unR)*(sM-unR)
 pStar=.5*(pStarL+pStarR)
 rSL=rhoL*(sL-unL)/(sL-sM)
 rSR=rhoR*(sR-unR)/(sR-sM)
 EL=((sL-unL)*L[3]-pL*unL+pStar*sM)/(sL-sM)
 ER=((sR-unR)*R[3]-pR*unR+pStar*sM)/(sR-sM)
 UstL=np.array([rSL,rSL*sM,rSL*utL,EL,L[4]*rSL/rhoL])
 UstR=np.array([rSR,rSR*sM,rSR*utR,ER,R[4]*rSR/rhoR])
 # Robust fallback: positivity/speed-invalid HLLC faces treated with Rusanov; record fallback count.
 invalid=(~np.isfinite(sM))|(~np.isfinite(pStar))|(pStar<=0)|(rSL<=0)|(rSR<=0)|(sL>sM)|(sM>sR)
 diss=np.maximum(np.abs(unL)+cL,np.abs(unR)+cR)
 fRus=.5*(fL+fR)-.5*diss[None,...]*(R-L)
 f=np.where((sL>=0)[None,...],fL,np.where((sM>=0)[None,...],fL+sL[None,...]*(UstL-L),np.where((sR>0)[None,...],fR+sR[None,...]*(UstR-R),fR)))
 f=np.where(invalid[None,...],fRus,f)
 if axis=='y': f=f[[0,2,1,3,4]]
 return f, int(np.count_nonzero(invalid))

def reconstruct(U,axis):
 s=slope_limited(U,1 if axis=='x' else 2)
 # Zhang-Shu style local admissibility safeguard: shrink reconstruction slope
 # until BOTH predicted cell-edge states have positive pressure and density.
 for _ in range(16):
  u0=U-.5*s;u1=U+.5*s
  bad=(u0[0]<=1e-9)|(u1[0]<=1e-9)
  for zz in (u0,u1):
   pp=(GAMMA-1)*(zz[3]-.5*(zz[1]**2+zz[2]**2)/np.maximum(zz[0],1e-12))
   bad|=(pp<=1e-9)
  if not np.any(bad):break
  s=np.where(bad[None,...],.5*s,s)
 else:raise RuntimeError('positivity reconstruction failed')
 if axis=='x':
  left=np.concatenate([U[:,0:1,:],U[:,:-1,:]+.5*s[:,:-1,:],U[:,-1:,:]],axis=1)
  right=np.concatenate([U[:,0:1,:],U[:,1:,:]-.5*s[:,1:,:],U[:,-1:,:]],axis=1)
 else:
  left=np.concatenate([U[:,:,0:1],U[:,:,:-1]+.5*s[:,:,:-1],U[:,:,-1:]],axis=2)
  right=np.concatenate([U[:,:,0:1],U[:,:,1:]-.5*s[:,:,1:],U[:,:,-1:]],axis=2)
 return left,right

def summary(U,X,Y,dx,dy):
    rho,vx,vy,p,a=primitives(U)
    W=np.maximum(U[4],0)
    norm=np.sum(W)+1e-14
    xmu=np.sum(W*X)/norm;ymu=np.sum(W*Y)/norm
    vxmu=np.sum(W*vx)/norm
    vymu=np.sum(W*(vy+Q*OMEGA*X))/norm
    Jmu=np.sum(W*(vy+2*OMEGA*X))/norm
    sx=np.sqrt(np.sum(W*(X-xmu)**2)/norm)
    sy=np.sqrt(np.sum(W*(Y-ymu)**2)/norm)
    vrel=np.sqrt(np.sum(W*(vx**2+(vy+Q*OMEGA*X)**2))/norm)
    return dict(tag_mass=float(np.sum(U[4])*dx*dy),rho_min=float(rho.min()),pressure_min=float(p.min()),
        rho_max=float(rho.max()),tag_centroid_x=float(xmu),tag_centroid_y=float(ymu),
        tag_width_x=float(sx),tag_width_y=float(sy),tag_mean_radial_v=float(vxmu),tag_mean_shear_subtracted_vy=float(vymu),
        tag_canonical_j=float(Jmu),tag_rms_peculiar_v=float(vrel))

from scipy.integrate import cumulative_trapezoid
C=1.022712165; GPC=4.30091e-3; G2=GPC*C*C
R0=5000.; M0=1.e8; c0=40.; rc=67.38314938927996; rt=171.16927720150684
MB,AB,MH,CH,BH=2.e9,700.,1.e11,12.,1.e6
rsm=239.36143006963653
VR,VP=-12.119420471965894,80.70517395981433
TMAX=4.; LX=120.; LY=440.
def func(x): return np.log1p(x)-x/(1+x)
def mhost(r):
 rsh=(3*MH/(800*np.pi*1.36e-7))**(1./3.)/CH
 return BH+MB*(r/(r+AB))**2+MH*func(r/rsh)/func(CH)
def vc(r):return np.sqrt(GPC*mhost(r)/r)
def hostacc(p):
 r=np.linalg.norm(p,axis=-1)
 return (-G2*mhost(r)/r**3)[...,None]*p
def orbitstep(p,v,dt):
 h=v+.5*dt*hostacc(p);q=p+dt*h
 return q,h+.5*dt*hostacc(q)
OMEGA=C*vc(R0)/R0
Q=1-(np.log(vc(R0+1))-np.log(vc(R0-1)))/(2./R0)
P0=1.380649e-16*5000/(1.3*1.67262192369e-24)/1e10*C*C
rgrid=np.geomspace(1e-5,rt,4000)
rhos=M0/(4*np.pi*rsm**3*func(c0))
rhodm=rhos/((rgrid/rsm)*(1+rgrid/rsm)**2)*np.exp(-rgrid/rc)/(1+rsm/rc)**.3
mgrid=np.r_[0,cumulative_trapezoid(4*np.pi*rgrid*rgrid*rhodm,rgrid)]
MASS_REL_ERROR=abs(mgrid[-1]/750692.8938805414-1)
assert MASS_REL_ERROR<1e-4
p=np.array([R0,0.]);v=C*np.array([VR,VP])
for i in range(200):p,v=orbitstep(p,v,-.01)
traj=[]; tt=np.arange(401)*.01-2
for i,t in enumerate(tt):
 angle=OMEGA*t;er=np.array([np.cos(angle),np.sin(angle)]);et=np.array([-er[1],er[0]])
 d=p-R0*er;traj.append([d@er,d@et])
 if i<400:p,v=orbitstep(p,v,.01)
traj=np.array(traj)
assert np.linalg.norm(traj[200])<.01

def initialize_box(nx,ny,Lx=LX,Ly=LY,phase=0):
 dx=Lx/nx;dy=Ly/ny
 X,Y=np.meshgrid(-Lx/2+(np.arange(nx)+.5)*dx,-Ly/2+(np.arange(ny)+.5)*dy,indexing='ij')
 B=np.exp(-.5*((X/18.)**2+((Y-phase)/70.)**2))
 S=np.exp(-.5*((X/3.)**2+((Y-phase)/35.)**2))
 rho=1+4*B+45*S;vx=np.zeros_like(X);vy=-Q*OMEGA*X
 return np.array([rho,rho*vx,rho*vy,P0/(GAMMA-1)+.5*rho*vy*vy,45*S]),X,Y,dx,dy

def stage_rhs(U,X,Y,dx,dy,t,b,speed,on,scheme):
 l,r=reconstruct(U,'x');fx,bx=hllc_flux(l,r,'x')
 l,r=reconstruct(U,'y');fy,by=hllc_flux(l,r,'y')
 d=-(fx[:,1:]-fx[:,:-1])/dx-(fy[:,:,1:]-fy[:,:,:-1])/dy
 rho,vx,vy,pressure,ss=primitives(U)
 ax=2*OMEGA*vy+2*Q*OMEGA*OMEGA*X;ay=-2*OMEGA*vx
 if on:
  dm_x=np.interp(t,tt,traj[:,0]);dm_y=np.interp(t,tt,traj[:,1])
  sx=X-dm_x;sy=Y-dm_y;rr=np.sqrt(sx*sx+sy*sy+.5**2)
  mm=np.interp(rr,rgrid,mgrid,left=0.,right=mgrid[-1])
  k=-G2*mm/rr**3;ax+=k*sx;ay+=k*sy
 d[1]+=rho*ax;d[2]+=rho*ay;d[3]+=rho*(vx*ax+vy*ay)
 return d,bx+by
def run_cp15(nx,ny,phase,on):
 U,X,Y,dx,dy=initialize_box(nx,ny,phase=phase)
 si=summary(U,X,Y,dx,dy);t=0.;n=0;fb=0
 while t<TMAX-1e-10:
  rho,vx,vy,p,a=primitives(U)
  dt=.25*min(dx/(np.max(abs(vx)+a)+1e-12),dy/(np.max(abs(vy)+a)+1e-12))
  dt=min(dt,TMAX-t)
  d1,f1=stage_rhs(U,X,Y,dx,dy,t-2,None,None,on,'HLLC')
  trial=U+dt*d1
  if trial[0].min()<=0 or primitives(trial)[3].min()<=1e-10:raise RuntimeError('predictor inadmissible')
  d2,f2=stage_rhs(trial,X,Y,dx,dy,t+dt-2,None,None,on,'HLLC')
  U=.5*(U+trial+dt*d2);t+=dt;n+=1;fb+=f1+f2
  if U[0].min()<=0 or primitives(U)[3].min()<=1e-10:raise RuntimeError('corrector inadmissible')
  if n>3000:raise RuntimeError('nonconvergent time loop')
 s=summary(U,X,Y,dx,dy)
 row=dict(nx=nx,ny=ny,phase_pc=phase,dm=bool(on),steps=n,fallback_faces=fb,
  cold_mean_vx_kms=s['tag_mean_radial_v']/C,cold_cx_pc=s['tag_centroid_x'],
  cold_mean_pecvy_kms=s['tag_mean_shear_subtracted_vy']/C,
  cold_width_x_pc=s['tag_width_x'],cold_mass_drift=(s['tag_mass']-si['tag_mass'])/si['tag_mass'],
  min_rho=s['rho_min'],min_p=s['pressure_min'],max_rho=s['rho_max'])
 print('case',nx,ny,phase,on,'mean_vx',round(row['cold_mean_vx_kms'],5),'steps',n,flush=True)
 return row,U




# CP19: neutral-atomic radiative HLLC proof-of-concept, unchanged halo orbit and hydro solver.
# Cooling source and pressure+thermally balanced endpoints from CP18.
import time
from hashlib import sha256
KB=1.380649e-16
MU=1.3; MP=1.67262192369e-24
SEC_PER_MYR=365.25*86400*1e6
GAMMA_HEAT=2e-26
PK=3000.
Tc=52.2963
Tw=6353.9480
nw=PK/Tw
nc=PK/Tc
RATIO=nc/nw
PC_PER_MYR_CGS=(1e5/C)
P0_RAD=KB*Tw/(MU*MP)/PC_PER_MYR_CGS**2
TMIN=5.0
COOL_A=2*SEC_PER_MYR/(3*KB) # K Myr^-1 per (erg s^-1)
# exact stable equilibrium roots refined from CP18 scipy
from scipy.optimize import brentq

def lam(T):
 T=np.maximum(T,1e-8)
 return GAMMA_HEAT*(1e7*np.exp(-118400./(T+1000.)) + .014*np.sqrt(T)*np.exp(-92./T))

def cool_equilibrium(pK, high=False):
 def f(T):return (pK/T)*float(lam(T))-GAMMA_HEAT
 return brentq(f, 3500.,10000.) if high else brentq(f, 10.,200.)
Tc=cool_equilibrium(PK);Tw=cool_equilibrium(PK,True)
nw=PK/Tw;nc=PK/Tc;RATIO=nc/nw
P0_RAD=KB*Tw/(MU*MP)/PC_PER_MYR_CGS**2
assert abs(nw*Tw-nc*Tc)<1e-9

def initialize_radiative(nx,ny,phase):
 dx=LX/nx;dy=LY/ny
 X,Y=np.meshgrid(-LX/2+(np.arange(nx)+.5)*dx,-LY/2+(np.arange(ny)+.5)*dy,indexing='ij')
 B=np.exp(-.5*((X/18.)**2+((Y-phase)/70.)**2))
 S=np.exp(-.5*((X/3.)**2+((Y-phase)/35.)**2))
 rho=1.+4.*B+(RATIO-5.)*S
 vy=-Q*OMEGA*X;vx=np.zeros_like(vy)
 tag=(RATIO-5.)*S
 U=np.array([rho,rho*vx,rho*vy,P0_RAD/(GAMMA-1)+.5*rho*vy**2,tag])
 return U,X,Y,dx,dy

def cooling_implicit(U,dt):
 # Isochoric (conservative cell fixed rho) backward Euler, exact root by 28 bisections;
 # Lambda(T) monotonic here => unique positive root.
 # T,new = T,old + dt * A (Gamma - n Lambda(T,new))
 if dt <=0:return U
 rho,vx,vy,p,a=primitives(U)
 old=Tw*(p/P0_RAD)/rho
 n=nw*rho
 low=np.full_like(old,TMIN)
 high=old+dt*COOL_A*GAMMA_HEAT
 # f(low)<0 and f(high)>=0 unless gas goes below floor; test this explicitly.
 def f(T):return T-old-dt*COOL_A*(GAMMA_HEAT-n*lam(T))
 if np.any(f(low)>0): raise RuntimeError('T floor activated, invalid cooling state')
 for _ in range(29):
  mid=(low+high)*.5
  move=f(mid)>0
  high=np.where(move,mid,high)
  low=np.where(move,low,mid)
 new=(low+high)*.5
 pn=rho*P0_RAD*(new/Tw)
 U[3]+= (pn-p)/(GAMMA-1)
 if not np.isfinite(U).all() or np.min(primitives(U)[3])<=0: raise RuntimeError('Cooling nonphysical')
 return U

def metrics_rad(U,X,Y,dx,dy):
 d=summary(U,X,Y,dx,dy)
 rho,vx,vy,p,a=primitives(U)
 T=Tw*(p/P0_RAD)/rho
 w=U[4]; total=w.sum()
 d.update(tag_mean_T_K=float((w*T).sum()/total),tag_fraction_T_below_100K=float(w[T<100].sum()/total),
          tag_fraction_T_above_300K=float(w[T>300].sum()/total),
          tag_rms_T_K=float(np.sqrt((w*(T-(w*T).sum()/total)**2).sum()/total)),
          T_min_K=float(T.min()),T_max_K=float(T.max()),
          tag_mean_specific_internal_energy_code=float(np.sum(w*(p/((GAMMA-1)*rho)))/total),
          tag_mean_entropy_proxy=float(np.sum(w*np.log(p/rho**GAMMA))/total))
 return d

def run_case(nx,ny,phase,with_dm,cooling=True,time_end=4.):
 U,X,Y,dx,dy=initialize_radiative(nx,ny,phase)
 first=metrics_rad(U,X,Y,dx,dy)
 t=0.;n=0;fb=0
 checkpoints=[]
 while t<time_end-1e-12:
  rho,vx,vy,p,a=primitives(U)
  dt=.25*min(dx/(np.max(np.abs(vx)+a)+1e-12),dy/(np.max(np.abs(vy)+a)+1e-12))
  dt=min(dt,time_end-t)
  if cooling: U=cooling_implicit(U,.5*dt)
  d1,ff1=stage_rhs(U,X,Y,dx,dy,t-2,None,None,with_dm,'HLLC')
  trial=U+dt*d1
  if trial[0].min()<=0 or primitives(trial)[3].min()<=1e-10:raise RuntimeError('predictor inadmissible')
  d2,ff2=stage_rhs(trial,X,Y,dx,dy,t+dt-2,None,None,with_dm,'HLLC')
  U=.5*(U+trial+dt*d2)
  if cooling:U=cooling_implicit(U,.5*dt)
  t+=dt;n+=1;fb+=ff1+ff2
  if n%10==0 or abs(t-time_end)<1e-10:
   m=metrics_rad(U,X,Y,dx,dy);m.update(time_Myr=float(t),phase_pc=phase,DM=int(with_dm),cooling=int(cooling),nx=nx,ny=ny)
   checkpoints.append(m)
  if n>1000:raise RuntimeError('Too many steps')
 end=metrics_rad(U,X,Y,dx,dy)
 row=dict(nx=nx,ny=ny,phase_pc=phase,dm=int(with_dm),cooling=int(cooling),steps=n,fallback_faces=fb,
    cold_mass_rel_drift=(end['tag_mass']/first['tag_mass']-1),
    tag_mean_T_K=end['tag_mean_T_K'],tag_fraction_T_below_100K=end['tag_fraction_T_below_100K'],
    tag_fraction_T_above_300K=end['tag_fraction_T_above_300K'],
    vphi_pec_kms=end['tag_mean_shear_subtracted_vy']/C,vr_kms=end['tag_mean_radial_v']/C,
    tag_centroid_pc=end['tag_centroid_x'],min_rho=end['rho_min'],min_pressure=end['pressure_min'],
    tag_log_entropy_proxy=end['tag_mean_entropy_proxy'],Tmin=end['T_min_K'],Tmax=end['T_max_K'])
 print('CASE',row,flush=True)
 return row,checkpoints,U



# DARK-ENTRAIN CP24: extended 2D radiative shearing-sheet finite-volume screening.
# ORIGINAL orbital profile / potential / cooling / Riemann solver are unchanged from CP22.
# Run many versions with CP24_GRID=... and CP24_PHASES=...; notebook auto packages outputs.
import os,hashlib,zipfile,time
CP24_VERSION='CP24_EXTENDED_HYDRO_2026-11-24'
LX=480.0;LY=1000.0
START=-18.0;END=22.0
CUTS24=(-40.,0.,40.)
SAMPLES=(-18.,-14.,-10.,-5.,0.,5.,10.,14.,18.,22.)
# Orbit from exactly the same CP23 static-host input, integrated symplectically.
def orbit_full_range(dt=.02):
 nneg=int(round(-START/dt));npos=int(round(END/dt)); pp=np.array([R0,0.]);vv=C*np.array([VR,VP]); arrneg=[]
 for j in range(nneg):
  pp,vv=orbitstep(pp,vv,-dt);arrneg.append(pp.copy())
 pp=np.array([R0,0.]);vv=C*np.array([VR,VP]);arrpos=[]
 for j in range(npos):
  pp,vv=orbitstep(pp,vv,dt);arrpos.append(pp.copy())
 tim=np.arange(-nneg,npos+1)*dt
 pos=np.array(arrneg[::-1]+[np.array([R0,0.])]+arrpos)
 angles=OMEGA*tim
 er=np.array([np.cos(angles),np.sin(angles)]).T
 et=np.array([-np.sin(angles),np.cos(angles)]).T
 delta=pos-R0*er
 track=np.stack([np.sum(delta*er,axis=1),np.sum(delta*et,axis=1)],axis=1)
 return tim,track,pos
full_tt,full_traj,full_pos=orbit_full_range()
# independent orbit timestep check and correct reference point.
other_tt,other_traj,other_pos=orbit_full_range(.01)
assert np.max(abs(full_traj-other_traj[::2]))<.005
assert np.max(abs(full_tt-other_tt[::2]))<1e-8
assert np.linalg.norm(full_traj[np.argmin(abs(full_tt))])<.001
# Use extended reference orbit, not a linear straight-line forced transit.
def rhs24(U,X,Y,dx,dy,t,with_dm):
 ll,rr=reconstruct(U,'x'); fx,bx=hllc_flux(ll,rr,'x')
 ll,rr=reconstruct(U,'y'); fy,by=hllc_flux(ll,rr,'y')
 d=-(fx[:,1:,:]-fx[:,:-1,:])/dx-(fy[:,:,1:]-fy[:,:,:-1])/dy
 rho,vx,vy,p,a=primitives(U)
 ax=2*OMEGA*vy+2*Q*OMEGA*OMEGA*X
 ay=-2*OMEGA*vx
 if with_dm:
  mx=np.interp(t,full_tt,full_traj[:,0]); my=np.interp(t,full_tt,full_traj[:,1])
  sx=X-mx;sy=Y-my; rr=np.sqrt(sx*sx+sy*sy+.5**2)
  mm=np.interp(rr,rgrid,mgrid,left=0.,right=mgrid[-1])
  a_dm=-G2*mm/(rr**3)
  ax+=a_dm*sx;ay+=a_dm*sy
 d[1]+=rho*ax;d[2]+=rho*ay;d[3]+=rho*(vx*ax+vy*ay)
 return d,fx[4],fy[4],bx+by

def integrate24(nx,ny,phase,dm,dtfactor=1.):
 # Tag is passive: conservative face flux exactly matches tracer continuity.
 U,X,Y,dx,dy=initialize_radiative(nx,ny,phase)
 if os.environ.get("CP24_TAG_COMPONENT", "core")=="envelope":
  B=gaussian_cell_avg(X,Y,dx,dy,18.,70.,phase)
  U[4]=4.*B
 rho,vx,vy,p,a=primitives(U)
 init_total=U[4].sum()*dx*dy
 faceidx={x:int(round((x+LX/2)/dx)) for x in CUTS24}
 assert all(abs((x+LX/2)/dx-i)<1e-8 for x,i in faceidx.items())
 assert all(0<i<nx for i in faceidx.values())
 acc={x:dict(cut=0.,left=0.,y_loss=0.) for x in CUTS24}
 startm={x:U[4,:i,:].sum()*dx*dy for x,i in faceidx.items()}
 t=START; n=0;fb=0; out=[]
 def snapshot():
  W=U[4];rr,vx,vy,pp,aa=primitives(U)
  tot=W.sum()*dx*dy
  d=dict(t_Myr=round(float(t),5),phase_pc=phase,DM=int(dm),nx=nx,ny=ny,
         delta_J=float((W*(vy+2*OMEGA*X)).sum()/W.sum()),
         tag_mean_vx_kms=float((W*vx).sum()/W.sum()/C),
         tag_center_x_pc=float((W*X).sum()/W.sum()),
         tag_mass_rel_change=float(tot/init_total-1.),
         rho_min=float(rr.min()),p_min=float(((GAMMA-1)*(U[3]-.5*(U[1]**2+U[2]**2)/U[0])).min()),tag_min=float(U[4].min()),tag_excess_max=float((U[4]-U[0]).max()),fallback_faces=fb,step_count=n)
  T=Tw*pp/(P0_RAD*rr); mx=np.interp(t,full_tt,full_traj[:,0]);my=np.interp(t,full_tt,full_traj[:,1]);R=np.hypot(X-mx,Y-my); cx=(W*X).sum()/tot;cy=(W*Y).sum()/tot
  cov=np.array([[(W*(X-cx)**2).sum(),(W*(X-cx)*(Y-cy)).sum()],[(W*(X-cx)*(Y-cy)).sum(),(W*(Y-cy)**2).sum()]])/tot
  eig=np.linalg.eigvalsh(cov);d.update(tag_mean_T_K=float((W*T).sum()/tot),tag_T_lt100=float(W[T<100].sum()/tot),tag_T_100_1000=float(W[(T>=100)&(T<1000)].sum()/tot),tag_T_ge1000=float(W[T>=1000].sum()/tot),tag_center_y_pc=float(cy),tag_rms_major_pc=float(np.sqrt(eig[1])),tag_rms_minor_pc=float(np.sqrt(eig[0])),tag_aspect=float(np.sqrt(eig[1]/eig[0])),tag_inside_20pc=float(W[R<20].sum()/init_total),tag_inside_40pc=float(W[R<40].sum()/init_total),tag_inside_rt=float(W[R<rt].sum()/init_total),dm_x_pc=float(mx),dm_y_pc=float(my),T_min_K=float(T.min()),T_max_K=float(T.max()))
  for label,mask in [('near',R<40),('far',R>=40)]:
   w=W[mask];d['tag_T_'+label+'_K']=float((w*T[mask]).sum()/w.sum()) if w.sum()>1e-12*init_total else float('nan')
  np.savez_compressed('frame_%+g_%d_%+06.1f.npz'%(phase,dm,t),U=U,x=x,y=y,t=t,phase=phase,dm=dm,dm_x=mx,dm_y=my)
  for cut,i in faceidx.items():
   cc='x'+str(int(cut));b=acc[cut]; m=W[:i,:].sum()*dx*dy
   d['cum_in_'+cc]=-b['cut']/init_total
   d['mass_incr_'+cc]=(m-startm[cut])/init_total
   d['budget_res_'+cc]=(m-startm[cut]-(b['left']-b['cut']-b['y_loss']))/init_total
   d['left_flux_'+cc]=b['left']/init_total
   d['y_loss_'+cc]=b['y_loss']/init_total
  return d
 out.append(snapshot())
 for target in SAMPLES[1:]:
  while t<target-1e-10:
   rho,vx,vy,p,a=primitives(U)
   dt=dtfactor*.22*min(dx/(np.max(np.abs(vx)+a)+1e-12),dy/(np.max(np.abs(vy)+a)+1e-12))
   dt=min(dt,target-t)
   U=cooling_implicit(U,.5*dt)
   d1,fx1,fy1,fb1=rhs24(U,X,Y,dx,dy,t,dm)
   mid=U+dt*d1
   if mid[0].min()<=0 or primitives(mid)[3].min()<=1e-9:raise RuntimeError('negative predictor '+str((t,phase,dm)))
   d2,fx2,fy2,fb2=rhs24(mid,X,Y,dx,dy,t+dt,dm)
   U=.5*(U+mid+dt*d2)
   U=cooling_implicit(U,.5*dt)
   if U[0].min()<=0 or primitives(U)[3].min()<=1e-9:raise RuntimeError('negative corrector '+str((t,phase,dm)))
   for cut,i in faceidx.items():
    b=acc[cut]
    b['cut']+=.5*dt*(float(fx1[i].sum())+float(fx2[i].sum()))*dy
    b['left']+=.5*dt*(float(fx1[0].sum())+float(fx2[0].sum()))*dy
    b['y_loss']+=.5*dt*(float((fy1[:i,-1]-fy1[:i,0]).sum())+float((fy2[:i,-1]-fy2[:i,0]).sum()))*dx
   n+=1;fb+=fb1+fb2;t+=dt
   if n>4000:raise RuntimeError('step limit')
  out.append(snapshot())
 maxres=max(abs(d['budget_res_x0']) for d in out)
 if maxres>1e-9:raise RuntimeError('budget failed '+str(maxres))
 print('CP24',nx,ny,phase,dm,'steps',n,'tag drift',out[-1]['tag_mass_rel_change'],'end flux in',out[-1]['cum_in_x0'],'max budget',maxres,flush=True)
 return out

def rows_csv(f,rr):
 with open(f,'w',newline='',encoding='utf-8') as g:
  w=csv.DictWriter(g,fieldnames=list(rr[0].keys()));w.writeheader();w.writerows(rr)

def go():
 import time
 tic=time.time()
 grid=int(os.environ.get('CP24_GRID','48'))
 if grid not in (48,72,96,120,144,160,192,240):raise ValueError('CP25_GRID must be 48,72,96,120,144,160,192,240')
 nx=grid;ny=grid*2
 phases=tuple(float(z) for z in os.environ.get('CP24_PHASES','-70,0,70').split(','))
 half=os.environ.get('CP24_HALF_DT','0')=='1'
 dts=.5 if half else 1.0
 rows=[]
 for phase in phases:
  for dm in (0,1):
   rows.extend(integrate24(nx,ny,phase,dm,dts))
 pairs=[]
 for phase in phases:
  for t in SAMPLES:
   aa=next(z for z in rows if z['phase_pc']==phase and z['DM']==1 and abs(z['t_Myr']-t)<1e-8)
   bb=next(z for z in rows if z['phase_pc']==phase and z['DM']==0 and abs(z['t_Myr']-t)<1e-8)
   d=dict(t_Myr=t,phase_pc=phase,nx=nx,ny=ny,
          delta_J=aa['delta_J']-bb['delta_J'],
          delta_tag_vr_kms=aa['tag_mean_vx_kms']-bb['tag_mean_vx_kms'],
          delta_center_x_pc=aa['tag_center_x_pc']-bb['tag_center_x_pc'])
   for x in CUTS24:
    cc='x'+str(int(x));d['delta_cum_in_'+cc]=aa['cum_in_'+cc]-bb['cum_in_'+cc]
    d['delta_mass_incr_'+cc]=aa['mass_incr_'+cc]-bb['mass_incr_'+cc]
   pairs.append(d)
 rows_csv('CP24_CASES.csv',rows);rows_csv('CP24_PAIRED_RESULTS.csv',pairs)
 with open('CP24_ORBIT_TIME.csv','w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['time_Myr','dm_x_pc','dm_y_pc','separation_to_gas_center_pc'])
  for it in range(0,len(full_tt),10):
   tt0=full_tt[it];xx,yy=full_traj[it];w.writerow([tt0,xx,yy,np.hypot(xx,yy)])
 # plot displayed physics from SAME actual simulation, with post-encounter shading
 plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,2,figsize=(11.6,8.4),constrained_layout=True)
 for ph in phases:
  pp=[x for x in pairs if x['phase_pc']==ph]; ts=np.array([x['t_Myr'] for x in pp]);
  lab=f'phase {ph:+.0f} pc'
  axs[0,0].plot(ts,[x['delta_cum_in_x0'] for x in pp],marker='o',label=lab)
  axs[0,1].plot(ts,[x['delta_cum_in_x-40'] for x in pp],marker='o',label=lab)
  axs[1,0].plot(ts,[x['delta_J'] for x in pp],marker='o',label=lab)
  axs[1,1].plot(ts,[x['delta_center_x_pc'] for x in pp],marker='o',label=lab)
 for a in axs.flat:
  a.axhline(0,color='gray',lw=.8,ls='--');a.axvspan(-13.7,13.7,color='gray',alpha=.06)
  a.set_xlabel('Time relative to nominal crossing (Myr)');a.legend(fontsize=8,loc='best')
 axs[0,0].set(ylabel='Excess cumulative inward cold-origin gas / initial tag mass',title='Conservative excess flux through x = 0')
 axs[0,1].set(ylabel='Excess cumulative inward fraction',title='Conservative excess flux through x = -40 pc')
 axs[1,0].set(ylabel='Excess mean J (pc / Myr)',title='Mean local canonical angular momentum difference')
 axs[1,1].set(ylabel='Gas-tag centroid difference (pc)',title='Difference in cold-origin gas centroid')
 fig.savefig('CP24_EXTENDED_TRANSPORT.png',dpi=170,bbox_inches='tight');plt.close(fig)
 report=dict(version=CP24_VERSION,domain=[LX,LY],start_Myr=START,end_Myr=END,grid=[nx,ny],phases=phases,
       cooling='Koyama Inutsuka simplified neutral atomic, implicit source same as CP19',
       limits=['Eulerian shearing sheet expanded to x±240 pc,y±500 pc, NOT a global disk simulation',
               'Cloud cold core width 3 pc underresolved on these grids; quantitative convergence cannot be asserted',
               'No self-consistent DM stripping or dynamical friction; 2D non-magnetic local model',
               'The fixed-box gas temperature transition is initially not in exact thermal equilibrium',
               'Matched DM-off controls and conservative flux necessary, not sufficient for astrophysical claim'],
       tagged_component=os.environ.get("CP24_TAG_COMPONENT", "core"),
       execution='Completed; wall time in run log only')
 Path('CP24_RUN_REPORT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 names=['CP24_CASES.csv','CP24_PAIRED_RESULTS.csv','CP24_ORBIT_TIME.csv','CP24_EXTENDED_TRANSPORT.png','CP24_RUN_REPORT.json']
 fixed=(2026,11,24,0,0,0)
 with zipfile.ZipFile('CP24_NOTEBOOK_AUTO_RESULTS.zip','w') as zz:
  for n in names:
   inf=zipfile.ZipInfo(n,fixed);inf.compress_type=zipfile.ZIP_DEFLATED
   zz.writestr(inf,Path(n).read_bytes())
 print('done elapsed',round(time.time()-tic,3),flush=True)


# CP25 finite-volume exact cell-integrated Gaussian IC, not a changed physical cloud.
from scipy.special import erf
from math import sqrt,pi

def one_dim_gaussian_avg(centers, step, sig, offset=0.):
    hh=step/(2.0*sqrt(2.)*sig)
    q=(centers-offset)/(sqrt(2.)*sig)
    return sqrt(pi/2.)*sig/step*(erf(q+hh)-erf(q-hh))

def gaussian_cell_avg(X,Y,dx,dy,sigx,sigy,phase):
    return one_dim_gaussian_avg(X,dx,sigx)*one_dim_gaussian_avg(Y,dy,sigy,phase)

def initialize_radiative(nx,ny,phase):
    dx=LX/nx;dy=LY/ny
    X,Y=np.meshgrid(-LX/2+(np.arange(nx)+.5)*dx,-LY/2+(np.arange(ny)+.5)*dy,indexing='ij')
    B=gaussian_cell_avg(X,Y,dx,dy,18.,70.,phase)
    S=gaussian_cell_avg(X,Y,dx,dy,3.,35.,phase)
    rho=1.+4.*B+(RATIO-5.)*S
    vy=-Q*OMEGA*X;vx=np.zeros_like(vy)
    tag=(RATIO-5.)*S
    U=np.array([rho,rho*vx,rho*vy,P0_RAD/(GAMMA-1)+.5*rho*vy**2,tag])
    return U,X,Y,dx,dy

CP24_VERSION = 'CP25_INITIAL_CONSERVATIVE_GAUSSIAN_AVERAGE_2026-11-24'


# CP27 static nonuniform mesh. Not AMR.
def grid26(h):
 x=np.unique(np.concatenate([np.linspace(-240,-120,13),np.linspace(-120,-80,21),np.linspace(-80,80,int(round(160/h))+1),np.linspace(80,120,21),np.linspace(120,240,13)]))
 y=np.unique(np.concatenate([np.linspace(-500,-180,33),np.linspace(-180,180,int(round(360/float(os.environ.get("CP29_DY","2.5"))))+1),np.linspace(180,500,33)]))
 return x,y

def rec26(U,e,axis):
 A=np.moveaxis(U,axis,1); w=np.diff(e);c=(e[1:]+e[:-1])/2
 s=np.zeros_like(A)
 s[:,1:-1]=_minmod((A[:,1:-1]-A[:,:-2])/np.diff(c)[:-1][None,:,None],(A[:,2:]-A[:,1:-1])/np.diff(c)[1:][None,:,None])*w[None,1:-1,None]
 for _ in range(16):
  lo=A-.5*s;hi=A+.5*s;bad=(lo[0]<=1e-9)|(hi[0]<=1e-9)
  for z in (lo,hi):bad|=((GAMMA-1)*(z[3]-.5*(z[1]**2+z[2]**2)/np.maximum(z[0],1e-12))<=1e-9)
  if not np.any(bad):break
  s=np.where(bad[None],.5*s,s)
 else:raise RuntimeError('bad reconstruction')
 L=np.concatenate([A[:,:1],A[:,:-1]+.5*s[:,:-1],A[:,-1:]],axis=1);R=np.concatenate([A[:,:1],A[:,1:]-.5*s[:,1:],A[:,-1:]],axis=1)
 return np.moveaxis(L,1,axis),np.moveaxis(R,1,axis)

def init26(x,y,phase):
 dx=np.diff(x)[:,None];dy=np.diff(y)[None,:];X,Y=np.meshgrid((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2,indexing='ij')
 B=gaussian_cell_avg(X,Y,dx,dy,18.,70.,phase);S=gaussian_cell_avg(X,Y,dx,dy,3.,35.,phase);rho=1+4*B+(RATIO-5)*S;vy=-Q*OMEGA*X
 return np.array([rho,np.zeros_like(rho),rho*vy,P0_RAD/(GAMMA-1)+.5*rho*vy**2,(RATIO-5)*S]),X,Y
def rhs26(U,X,Y,x,y,t,with_dm):
 ll,rr=rec26(U,x,1); fx,bx=hllc_flux(ll,rr,'x')
 ll,rr=rec26(U,y,2); fy,by=hllc_flux(ll,rr,'y')
 d=-(fx[:,1:,:]-fx[:,:-1,:])/np.diff(x)[None,:,None]-(fy[:,:,1:]-fy[:,:,:-1])/np.diff(y)[None,None,:]
 rho,vx,vy,p,a=primitives(U)
 ax=2*OMEGA*vy+2*Q*OMEGA*OMEGA*X
 ay=-2*OMEGA*vx
 if with_dm:
  mx=np.interp(t,full_tt,full_traj[:,0]); my=np.interp(t,full_tt,full_traj[:,1])
  sx=X-mx;sy=Y-my; rr=np.sqrt(sx*sx+sy*sy+.5**2)
  mm=np.interp(rr,rgrid,mgrid,left=0.,right=mgrid[-1])
  a_dm=-G2*mm/(rr**3)
  ax+=a_dm*sx;ay+=a_dm*sy
 d[1]+=rho*ax;d[2]+=rho*ay;d[3]+=rho*(vx*ax+vy*ay)
 return d,fx[4],fy[4],bx+by

def integrate26(h,phase,dm,dtfactor=1.,end=22.):
 # Tag is passive: conservative face flux exactly matches tracer continuity.
 x,y=grid26(h);U,X,Y=init26(x,y,phase);dx=np.diff(x);dy=np.diff(y);vol=dx[:,None]*dy[None,:];nx,ny=X.shape
 if os.environ.get("CP24_TAG_COMPONENT", "core")=="envelope":
  B=gaussian_cell_avg(X,Y,dx[:,None],dy[None,:],18.,70.,phase)
  U[4]=4.*B
 rho,vx,vy,p,a=primitives(U)
 init_total=(U[4]*vol).sum()
 faceidx={c:int(np.where(np.isclose(x,c))[0][0]) for c in CUTS24}
 assert all(0<i<nx for i in faceidx.values())
 acc={x:dict(cut=0.,left=0.,y_loss=0.) for x in CUTS24}
 startm={x:(U[4,:i,:]*vol[:i]).sum() for x,i in faceidx.items()}
 t=START; n=0;fb=0; out=[]
 def snapshot():
  W=U[4]*vol;rr,vx,vy,pp,aa=primitives(U)
  tot=W.sum()
  d=dict(t_Myr=round(float(t),5),phase_pc=phase,DM=int(dm),nx=nx,ny=ny,
         delta_J=float((W*(vy+2*OMEGA*X)).sum()/W.sum()),
         tag_mean_vx_kms=float((W*vx).sum()/W.sum()/C),
         tag_center_x_pc=float((W*X).sum()/W.sum()),
         tag_mass_rel_change=float(tot/init_total-1.),
         rho_min=float(rr.min()),p_min=float(((GAMMA-1)*(U[3]-.5*(U[1]**2+U[2]**2)/U[0])).min()),tag_min=float(U[4].min()),tag_excess_max=float((U[4]-U[0]).max()),fallback_faces=fb,step_count=n)
  T=Tw*pp/(P0_RAD*rr); mx=np.interp(t,full_tt,full_traj[:,0]);my=np.interp(t,full_tt,full_traj[:,1]);R=np.hypot(X-mx,Y-my); cx=(W*X).sum()/tot;cy=(W*Y).sum()/tot
  cov=np.array([[(W*(X-cx)**2).sum(),(W*(X-cx)*(Y-cy)).sum()],[(W*(X-cx)*(Y-cy)).sum(),(W*(Y-cy)**2).sum()]])/tot
  eig=np.linalg.eigvalsh(cov);d.update(tag_mean_T_K=float((W*T).sum()/tot),tag_T_lt100=float(W[T<100].sum()/tot),tag_T_100_1000=float(W[(T>=100)&(T<1000)].sum()/tot),tag_T_ge1000=float(W[T>=1000].sum()/tot),tag_center_y_pc=float(cy),tag_rms_major_pc=float(np.sqrt(eig[1])),tag_rms_minor_pc=float(np.sqrt(eig[0])),tag_aspect=float(np.sqrt(eig[1]/eig[0])),tag_inside_20pc=float(W[R<20].sum()/init_total),tag_inside_40pc=float(W[R<40].sum()/init_total),tag_inside_rt=float(W[R<rt].sum()/init_total),dm_x_pc=float(mx),dm_y_pc=float(my),T_min_K=float(T.min()),T_max_K=float(T.max()))
  for label,mask in [('near',R<40),('far',R>=40)]:
   w=W[mask];d['tag_T_'+label+'_K']=float((w*T[mask]).sum()/w.sum()) if w.sum()>1e-12*init_total else float('nan')
  np.savez_compressed('frame_%+g_%d_%+06.1f.npz'%(phase,dm,t),U=U,x=x,y=y,t=t,phase=phase,dm=dm,dm_x=mx,dm_y=my)
  for cut,i in faceidx.items():
   cc='x'+str(int(cut));b=acc[cut]; m=W[:i,:].sum()
   d['cum_in_'+cc]=-b['cut']/init_total
   d['mass_incr_'+cc]=(m-startm[cut])/init_total
   d['budget_res_'+cc]=(m-startm[cut]-(b['left']-b['cut']-b['y_loss']))/init_total
   d['left_flux_'+cc]=b['left']/init_total
   d['y_loss_'+cc]=b['y_loss']/init_total
  return d
 key='h%g_phase%+g_dm%d'%(h,phase,dm)
 if Path(key+'_state.npz').exists() and Path(key+'.csv').exists():
  z=np.load(key+'_state.npz');assert np.array_equal(z['x'],x) and np.array_equal(z['y'],y);U=z['U'];t=float(z['t']);n=int(z['n']);fb=int(z['fb']);acc={c:dict(zip(('cut','left','y_loss'),z['acc'][j])) for j,c in enumerate(CUTS24)}
  with open(key+'.csv') as f:out=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
  assert abs(out[-1]['t_Myr']-t)<1e-8
 else:out.append(snapshot())
 for target in sorted(set([s for s in SAMPLES if t<s<=end]+[end])):
  if target<=t+1e-10:continue
  while t<target-1e-10:
   rho,vx,vy,p,a=primitives(U)
   dt=dtfactor*.22*min(np.min(dx[:,None]/(np.abs(vx)+a+1e-12)),np.min(dy[None,:]/(np.abs(vy)+a+1e-12)))
   dt=min(dt,target-t)
   U=cooling_implicit(U,.5*dt)
   d1,fx1,fy1,fb1=rhs26(U,X,Y,x,y,t,dm)
   mid=U+dt*d1
   if mid[0].min()<=0 or primitives(mid)[3].min()<=1e-9:raise RuntimeError('negative predictor '+str((t,phase,dm)))
   d2,fx2,fy2,fb2=rhs26(mid,X,Y,x,y,t+dt,dm)
   U=.5*(U+mid+dt*d2)
   U=cooling_implicit(U,.5*dt)
   if U[0].min()<=0 or primitives(U)[3].min()<=1e-9:raise RuntimeError('negative corrector '+str((t,phase,dm)))
   for cut,i in faceidx.items():
    b=acc[cut]
    b['cut']+=.5*dt*float(((fx1[i]+fx2[i])*dy).sum())
    b['left']+=.5*dt*float(((fx1[0]+fx2[0])*dy).sum())
    b['y_loss']+=.5*dt*float(((fy1[:i,-1]-fy1[:i,0]+fy2[:i,-1]-fy2[:i,0])*dx[:i]).sum())
   n+=1;fb+=fb1+fb2;t+=dt
   if n>30000:raise RuntimeError('step limit')
  out.append(snapshot())
  rows_csv(key+'.csv',out);np.savez_compressed(key+'_state.npz',U=U,x=x,y=y,t=t,n=n,fb=fb,acc=np.array([[acc[c][k] for k in ('cut','left','y_loss')] for c in CUTS24]));print(key,t,n,flush=True)
 maxres=max(abs(d['budget_res_x'+str(int(c))]) for d in out for c in CUTS24)
 if maxres>1e-9:raise RuntimeError('budget failed '+str(maxres))
 print('CP24',nx,ny,phase,dm,'steps',n,'tag drift',out[-1]['tag_mass_rel_change'],'end flux in',out[-1]['cum_in_x0'],'max budget',maxres,flush=True)
 return out


def validate26():
 x=np.linspace(-240,240,73);y=np.linspace(-500,500,145);U,X,Y=init26(x,y,70)
 d,*_=rhs26(U,X,Y,x,y,0,True);D,*_=rhs24(U,X,Y,np.diff(x)[0],np.diff(y)[0],0,True);err=float(np.max(abs(d-D)));assert err<1e-8
 x,y=grid26(1);U,X,Y=init26(x,y,70);dx=np.diff(x);dy=np.diff(y);vol=dx[:,None]*dy[None,:]
 d,f,g,b=rhs26(U,X,Y,x,y,0,True);res=(d[4]*vol).sum()+((f[-1]-f[0])*dy).sum()+((g[:,-1]-g[:,0])*dx).sum();mass=(U[4]*vol).sum()/((RATIO-5)*2*np.pi*3*35)-1
 L=np.stack([2+X*.001+Y*.0001]*5);L[1:3]=0;L[3]=10;L[4]=L[0]*.2;A,B=rec26(L,x,1);linear=float(np.max(abs(A[0,2:-2]-(2+x[2:-2,None]*.001+Y[:1]*.0001))))
 assert abs(res)<1e-8 and abs(mass)<1e-12 and linear<1e-12
 v=dict(uniform_rhs_max_error=err,nonuniform_tracer_rhs_residual=float(res),initial_mass_relative_error=float(mass),linear_reconstruction_error=linear);Path('CP27_VALIDATION.json').write_text(json.dumps(v,indent=2));print(v,flush=True)


# Same backward Euler root, safeguarded Newton with original bisection fallback.
_original_cooling=cooling_implicit
def cooling_implicit(U,dt):
 if dt<=0:return U
 rho,vx,vy,p,a=primitives(U);old=Tw*p/(P0_RAD*rho); n=nw*rho; A=dt*COOL_A*GAMMA_HEAT
 lo=np.full_like(old,TMIN);hi=old+A; z=np.clip(old,lo,hi)
 def F(T):return T-old-A*(1-n*(1e7*np.exp(-118400/(T+1000))+.014*np.sqrt(T)*np.exp(-92/T)))
 if np.any(F(lo)>0):raise RuntimeError('temperature floor')
 for i in range(16):
  L1=1e7*np.exp(-118400/(z+1000));L2=.014*np.sqrt(z)*np.exp(-92/z); f=z-old-A*(1-n*(L1+L2));der=1+A*n*(L1*118400/(z+1000)**2+L2*(.5/z+92/z**2))
  if np.max(np.abs(f)/(1+old+A))<2e-12:break
  lo=np.where(f<0,z,lo);hi=np.where(f>0,z,hi);new=z-f/der
  new=np.where((new>=lo)&(new<=hi),new,.5*(lo+hi));z=np.where(np.abs(f)/(1+old+A)<2e-12,z,new)
 else:return _original_cooling(U,dt)
 U[3]+=(rho*P0_RAD*z/Tw-p)/(GAMMA-1);return U

def validate29():
 validate26();rng=np.random.default_rng(29);r=10**rng.uniform(-.1,2.5,(48,48));T=10**rng.uniform(1.5,4.4,r.shape);z=np.zeros_like(r);u=np.array([r,z,z,r*P0_RAD*T/Tw/(GAMMA-1),z]);errors=[]
 for dt in (.001,.005,.01,.025):
  a=cooling_implicit(u.copy(),dt);b=_original_cooling(u.copy(),dt);errors.append(float(np.max(np.abs(a[3]-b[3])/b[3])))
 assert max(errors)<2e-6,errors
 Path('CP29_COOLING_VALIDATION.json').write_text(json.dumps({'relative_energy_error_vs_bisection':errors},indent=2))

def main29():
 validate29();phases=[float(v) for v in os.environ.get('CP29_PHASES','70,-70').split(',')];end=float(os.environ.get('CP29_END','22'));dmodes=[int(os.environ['CP29_DM'])] if 'CP29_DM' in os.environ else [0,1]
 for p in phases:
  for dm in dmodes:integrate26(1.,p,dm,end=end)
 with zipfile.ZipFile('CP29_AUTO_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(Path('.').glob('h*_phase*'))+sorted(Path('.').glob('frame_*.npz'))+sorted(Path('.').glob('CP*_VALIDATION.json')):z.write(p,p.name)
if __name__=='__main__':main29()
