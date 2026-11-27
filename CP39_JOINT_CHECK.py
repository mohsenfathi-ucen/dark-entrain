"""Post-process saved full encounter without advancing the hydro solver.
Direct tagged mass intersections, fixed initial mass denominator, independent angular momentum check.
"""
from pathlib import Path
import csv, math
import numpy as np
import CP29_SOURCE as s
ROOT=Path(__file__).resolve().parent
# Regenerate prescribed slow trajectory from passage state, independent of remapping.
s.START=-50.; s.END=50.; s.VR*=.5
times,track,_=s.orbit_full_range()
kappa2=2*(2-s.Q)*s.OMEGA**2
rows=[]
for dm in (0,1):
 paths=sorted((ROOT/f'full_DM{dm}').glob('frame_*.npz'))
 if not paths: raise RuntimeError('Missing snapshots')
 fs=[]
 for p in paths:
  with np.load(p) as z: fs.append((float(z['t']),p))
 fs.sort()
 for t,p in fs:
  with np.load(p) as z: U=z['U']; x=z['x']; y=z['y']
  X=(x[:-1]+x[1:])[:,None]/2; Y=(y[:-1]+y[1:])[None,:]/2
  W=U[4]*np.diff(x)[:,None]*np.diff(y)[None,:]; mass=W.sum()
  if t==fs[0][0]: m0=mass
  assert abs(mass/m0-1)<1e-12 and np.isfinite(U).all()
  vx=U[1]/U[0]; vy=U[2]/U[0]; j=vy+2*s.OMEGA*X
  mx,my=[np.interp(t,times,track[:,i]) for i in (0,1)]
  R=np.hypot(X-mx,Y-my); I=np.broadcast_to(X<-40,W.shape); J=j>0
  f=lambda mask: float(W[mask].sum()/m0)
  mean=lambda field: float(np.sum(W*field)/mass)
  a,b,c=f(I),f(I&(R>=40)),f(I&J)
  triple=f(I&(R>=40)&J); triple_rt=f(I&(R>=s.rt)&J)
  assert max(0,b+c-a)-1e-12<=triple<=min(b,c)+1e-12
  assert abs(mean(-kappa2*(X-j/((2-s.Q)*s.OMEGA)))-mean(2*s.OMEGA*vy+2*s.Q*s.OMEGA**2*X))<1e-12
  rows.append(dict(dm=dm,t_Myr=t,inner=a,inner_far40=b,inner_positive_j=c,inner_far40_positive_j=triple,inner_far_rt_positive_j=triple_rt,mean_free_radial_accel=mean(2*s.OMEGA*vy+2*s.Q*s.OMEGA**2*X),mean_x=mean(X),mean_j=mean(j),mean_vx=mean(vx),period_Myr=2*math.pi/math.sqrt(kappa2)))
with (ROOT/'CP39_JOINT_CHECK.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if len(rows)==42:
 with (ROOT/'CP38_KINEMATICS.csv').open() as f: old={(int(r['dm']),float(r['t_Myr'])):r for r in csv.DictReader(f)}
 with (ROOT/'CP37_FIELDS.csv').open() as f: old2={(int(r['dm']),float(r['t_Myr'])):r for r in csv.DictReader(f)}
 for r in rows:
  key=(r['dm'],r['t_Myr']); assert abs(r['inner_positive_j']-float(old[key]['inner_positive_xg']))<1e-11; assert abs(r['inner_far40']-float(old2[key]['inner_far40']))<1e-11
for r in rows:
 if r['t_Myr']==50: print(r)
