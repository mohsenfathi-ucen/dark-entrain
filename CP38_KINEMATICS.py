"""Cell-area-weighted instantaneous kinematics; no prediction of future trajectories."""
from pathlib import Path
import numpy as np, csv, json
ROOT=Path(__file__).resolve().parent
# Use the same analytic host parameters and physical units as production.
import CP29_SOURCE as s
rows=[]
for dm in (0,1):
 for p in sorted((ROOT/f'full_DM{dm}').glob('frame_*.npz')):
  with np.load(p) as z:
   U=z['U']; x=z['x']; y=z['y']; t=float(z['t'])
  X=(x[:-1]+x[1:])[:,None]/2; A=np.diff(x)[:,None]*np.diff(y)[None,:]; W=U[4]*A; M=W.sum(); vx=U[1]/U[0]; vy=U[2]/U[0]; j=vy+2*s.OMEGA*X; xg=j/((2-s.Q)*s.OMEGA)
  avg=lambda v: float(np.sum(W*v)/M)
  row=dict(dm=dm,t_Myr=t,x_pc=avg(X),vx_pc_Myr=avg(vx),j_pc_Myr=avg(j),xg_pc=avg(xg),x_minus_xg_pc=avg(X-xg),inner_fraction=avg(X<-40),inner_positive_xg=avg((X<-40)&(xg>0)),inner_inward_velocity=avg((X<-40)&(vx<0)),free_epicyclic_acceleration_pc_Myr2=avg(-2*(2-s.Q)*s.OMEGA**2*(X-xg)))
  assert abs(row['xg_pc']-row['j_pc_Myr']/((2-s.Q)*s.OMEGA))<1e-10
  rows.append(row)
rows.sort(key=lambda r:(r['dm'],r['t_Myr']))
with (ROOT/'CP38_KINEMATICS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Compare the separately archived diagnostics when the full output is present.
with (ROOT/'CP37_FIELDS.csv').open() as f: old={(int(r['dm']),float(r['t_Myr'])):r for r in csv.DictReader(f)}
for r in rows:
 a=old.get((r['dm'],r['t_Myr']))
 if a:
  assert abs(r['j_pc_Myr']-float(a['J']))<1e-10
  assert abs(r['x_pc']-float(a['centroid']))<1e-10
print(json.dumps([r for r in rows if r['t_Myr'] in (20,30,40,50)],indent=2))
