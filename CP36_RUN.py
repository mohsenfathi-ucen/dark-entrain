"""Independent full-duration wide-strip pair; no restart from CP35."""
import sys, os, json
from pathlib import Path
import numpy as np
import CP29_SOURCE as s
ROOT=Path(__file__).resolve().parent
if os.environ.get('CP36_NUMPY')!='1':
 try:
  import recovery_fast as fast
  s.rhs26=fast.rhs; s.cooling_implicit=fast.cool
 except ImportError: pass

def setup():
 with np.load(ROOT/'GRID.npz') as g: x=g['x']; y=g['y']
 s.START=-50.;s.END=50.;s.VR*=.5;s.full_tt,s.full_traj,s.full_pos=s.orbit_full_range()
 s.SAMPLES=tuple(np.arange(-50,51,5.));s.grid26=lambda h:(x,y)
 return x,y
if __name__=='__main__':
 dm=int(sys.argv[1]);x,y=setup();U,X,Y=s.init26(x,y,70.)
 Path('configuration.json').write_text(json.dumps(dict(dm=dm,start=-50,end=50,mesh=list(X.shape),initialization='analytic cell averages',min_dx=float(np.diff(x).min()),min_dy=float(np.diff(y).min()),initial_mass=float(np.sum(U[4]*np.diff(x)[:,None]*np.diff(y)[None,:]))),indent=2))
 s.integrate26(1.,70.,dm,end=float(os.environ.get('CP36_END','50')))
