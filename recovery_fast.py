"""Compiled scalar-loop backend for recovered CP29 discretization. No fastmath."""
import numpy as np
from numba import njit
import CP29_SOURCE as s
G=s.GAMMA
@njit(inline='always')
def p(r,m,n,e):return (G-1)*(e-.5*(m*m+n*n)/max(r,1e-12))
@njit(cache=True)
def slopes(U,e):
 n,m=U.shape[1:]; S=np.zeros_like(U)
 for i in range(1,n-1):
  w=e[i+1]-e[i]; a=.5*(e[i+1]-e[i-1]);b=.5*(e[i+2]-e[i])
  for j in range(m):
   for k in range(5):
    l=(U[k,i,j]-U[k,i-1,j])/a;r=(U[k,i+1,j]-U[k,i,j])/b
    if l*r>0:S[k,i,j]=np.sign(l)*min(abs(l),abs(r))*w
   for k in range(16):
    r=U[0,i,j];mx=U[1,i,j];my=U[2,i,j];en=U[3,i,j]
    dr=.5*S[0,i,j];dx=.5*S[1,i,j];dy=.5*S[2,i,j];de=.5*S[3,i,j]
    if min(r-dr,r+dr)>1e-9 and min(p(r-dr,mx-dx,my-dy,en-de),p(r+dr,mx+dx,my+dy,en+de))>1e-9:break
    for v in range(5):S[v,i,j]*=.5
   else:raise ValueError('reconstruction')
 return S
@njit(cache=True)
def faces(U,e):
 n,m=U.shape[1:];S=slopes(U,e);F=np.empty((5,n+1,m));nb=0
 for i in range(n+1):
  il=max(0,i-1);ir=min(n-1,i)
  for j in range(m):
   rl=U[0,il,j]+.5*S[0,il,j];rr=U[0,ir,j]-.5*S[0,ir,j]
   ml=U[1,il,j]+.5*S[1,il,j];mr=U[1,ir,j]-.5*S[1,ir,j]
   nl=U[2,il,j]+.5*S[2,il,j];nr=U[2,ir,j]-.5*S[2,ir,j]
   el=U[3,il,j]+.5*S[3,il,j];er=U[3,ir,j]-.5*S[3,ir,j]
   wl=U[4,il,j]+.5*S[4,il,j];wr=U[4,ir,j]-.5*S[4,ir,j]
   pl=p(rl,ml,nl,el);pr=p(rr,mr,nr,er)
   if min(rl,rr,pl,pr)<=0:raise ValueError('bad face')
   ul=ml/rl;ur=mr/rr;vl=nl/rl;vr=nr/rr;cl=np.sqrt(G*pl/rl);cr=np.sqrt(G*pr/rr)
   sl=min(ul-cl,ur-cr);sr=max(ul+cl,ur+cr)
   sm=(pr-pl+rl*ul*(sl-ul)-rr*ur*(sr-ur))/(rl*(sl-ul)-rr*(sr-ur))
   ps=.5*((pl+rl*(sl-ul)*(sm-ul))+(pr+rr*(sr-ur)*(sm-ur)))
   rsl=rl*(sl-ul)/(sl-sm);rsr=rr*(sr-ur)/(sr-sm)
   fl=(ml,ml*ul+pl,nl*ul,(el+pl)*ul,wl*ul);fr=(mr,mr*ur+pr,nr*ur,(er+pr)*ur,wr*ur)
   L=(rl,ml,nl,el,wl);R=(rr,mr,nr,er,wr)
   if not np.isfinite(sm) or not np.isfinite(ps) or ps<=0 or rsl<=0 or rsr<=0 or sl>sm or sm>sr:
    diss=max(abs(ul)+cl,abs(ur)+cr);nb+=1
    for k in range(5):F[k,i,j]=.5*(fl[k]+fr[k])-.5*diss*(R[k]-L[k])
   elif sl>=0:
    for k in range(5):F[k,i,j]=fl[k]
   elif sm>=0:
    es=((sl-ul)*el-pl*ul+ps*sm)/(sl-sm);st=(rsl,rsl*sm,rsl*vl,es,wl*rsl/rl)
    for k in range(5):F[k,i,j]=fl[k]+sl*(st[k]-L[k])
   elif sr>0:
    es=((sr-ur)*er-pr*ur+ps*sm)/(sr-sm);st=(rsr,rsr*sm,rsr*vr,es,wr*rsr/rr)
    for k in range(5):F[k,i,j]=fr[k]+sr*(st[k]-R[k])
   else:
    for k in range(5):F[k,i,j]=fr[k]
 return F,nb
@njit(cache=True)
def cool_kernel(U,dt,Tw,P0,nw,ca,gh,TMIN):
 A=dt*ca*gh
 for i in range(U.shape[1]):
  for j in range(U.shape[2]):
   r=U[0,i,j];p0=max(p(r,U[1,i,j],U[2,i,j],U[3,i,j]),1e-8);old=Tw*p0/(P0*r);n=nw*r
   lo=TMIN;hi=old+A;z=min(max(old,lo),hi)
   for k in range(16):
    L1=1e7*np.exp(-118400/(z+1000));L2=.014*np.sqrt(z)*np.exp(-92/z);f=z-old-A*(1-n*(L1+L2));der=1+A*n*(L1*118400/(z+1000)**2+L2*(.5/z+92/z**2))
    if abs(f)/(1+old+A)<2e-12:break
    if f<0:lo=z
    elif f>0:hi=z
    new=z-f/der;z=new if lo<=new<=hi else .5*(lo+hi)
   else:return False
   U[3,i,j]+=(r*P0*z/Tw-p0)/(G-1)
 return True
orig_cooling=s.cooling_implicit
orig_rhs=s.rhs26
def cool(U,dt):
 V=U.copy()
 if cool_kernel(V,dt,s.Tw,s.P0_RAD,s.nw,s.COOL_A,s.GAMMA_HEAT,s.TMIN):return V
 return orig_cooling(U,dt)
def rhs(U,X,Y,x,y,t,with_dm):
 fx,bx=faces(U,x);fy,by=faces(np.ascontiguousarray(U[[0,2,1,3,4]].transpose(0,2,1)),y);fy=fy[[0,2,1,3,4]].transpose(0,2,1)
 d=-(fx[:,1:,:]-fx[:,:-1,:])/np.diff(x)[None,:,None]-(fy[:,:,1:]-fy[:,:,:-1])/np.diff(y)[None,None,:]
 rho,vx,vy,pr,a=s.primitives(U);ax=2*s.OMEGA*vy+2*s.Q*s.OMEGA*s.OMEGA*X;ay=-2*s.OMEGA*vx
 if with_dm:
  mx=np.interp(t,s.full_tt,s.full_traj[:,0]);my=np.interp(t,s.full_tt,s.full_traj[:,1]);sx=X-mx;sy=Y-my;rr=np.sqrt(sx*sx+sy*sy+.5**2);mm=np.interp(rr,s.rgrid,s.mgrid,left=0.,right=s.mgrid[-1]);aa=-s.G2*mm/(rr**3);ax+=aa*sx;ay+=aa*sy
 d[1]+=rho*ax;d[2]+=rho*ay;d[3]+=rho*(vx*ax+vy*ay)
 return d,fx[4],fy[4],bx+by
