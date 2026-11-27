"""Recheck saved CP37 fields using recorded trajectory and independent masked quadrature; no hydro."""
import csv, math, json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
old={(int(r['dm']),float(r['t_Myr'])):r for r in csv.DictReader((root/'CP39_JOINT_CHECK.csv').open())}
# Use recorded trajectory and infer the exact Omega from archived period and q (rounded q below is not used).
import CP29_SOURCE as s
omega=s.OMEGA
results=[]; maxdiff=0; maxdrift=0
for dm in (0,1):
 folder=root/f'full_DM{dm}'
 hist={float(r['t_Myr']):r for r in csv.DictReader(next(folder.glob('h1_phase*.csv')).open())}
 frames=[]
 for f in folder.glob('frame_*.npz'):
  with np.load(f) as z: frames.append((float(z['t']),f))
 for t,f in sorted(frames):
  with np.load(f) as z: U=z['U']; x=z['x']; y=z['y']
  rho,mx,my,E,tag=U
  area=np.outer(np.diff(x),np.diff(y)); weights=area*tag
  mass=math.fsum(weights.ravel());
  if t==-50: m0=mass
  maxdrift=max(maxdrift,abs(mass/m0-1))
  P=(5/3-1)*(E-(mx*mx+my*my)/(2*rho))
  assert np.isfinite(U).all() and rho.min()>0 and P.min()>0 and tag.min()>=-1e-12 and (tag-rho).max()<=1e-12
  X,Y=np.meshgrid((x[:-1]+x[1:])/2,(y[:-1]+y[1:])/2,indexing='ij')
  # Determine sign without dividing by rho; use squared distance rather than hypot.
  jpositive=my+2*omega*X*rho>0
  h=hist[t]; R2=(X-float(h['dm_x_pc']))**2+(Y-float(h['dm_y_pc']))**2
  I=X<-40
  frac=lambda mask: math.fsum(weights[mask].ravel())/m0
  vals={'inner':frac(I),'inner_far40':frac(I&(R2>=1600)), 'inner_positive_j':frac(I&jpositive),'inner_far40_positive_j':frac(I&jpositive&(R2>=1600)), 'inner_far_rt_positive_j':frac(I&jpositive&(R2>=s.rt**2))}
  for k,v in vals.items(): maxdiff=max(maxdiff,abs(v-float(old[(dm,t)][k])))
  results.append(dict(dm=dm,t=t,**vals))
assert len(results)==42 and maxdiff<1e-11 and maxdrift<1e-12
out={'frames':len(results),'max_fraction_difference':maxdiff,'max_mass_drift':maxdrift,'all_positive_density_pressure_and_bounded_tag':True,'endpoint':[r for r in results if r['t']==50]}
(root/'CP40_VERIFY_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
