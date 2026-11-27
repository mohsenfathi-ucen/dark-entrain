"""Independent snapshot checks, paired comparisons, and shape-preserving display.
No extrapolation beyond the last common saved time. Uses archived CP31 and CP35 tables.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import CP29_SOURCE as s
from CP36_RUN import setup
root=Path(__file__).resolve().parent
setup()
ot=s.full_tt; op=s.full_traj; ov=np.gradient(op,ot,axis=0)
r=np.geomspace(.5,s.rt,40000); inte=cumulative_trapezoid(np.interp(r,s.rgrid,s.mgrid)/r**2,r,initial=0); M=s.mgrid[-1]; pot=-s.G2*(M/s.rt+inte[-1]-inte)
rows=[]; comparisons=[]
for dm in (0,1):
    folder=root/f'full_DM{dm}'; d=pd.read_csv(folder/f'h1_phase+70_dm{dm}.csv').sort_values('t_Myr'); old=pd.read_csv(root/f'original_DM{dm}.csv'); init=None; start={}
    for _,a in d.iterrows():
        t=float(a.t_Myr)
        with np.load(folder/f'frame_+70_{dm}_{t:+06.1f}.npz') as z:
            U=z['U'];x=z['x'];y=z['y']; assert abs(float(z['t'])-t)<1e-8
        X,Y=np.meshgrid((x[:-1]+x[1:])/2,(y[:-1]+y[1:])/2,indexing='ij'); A=np.diff(x)[:,None]*np.diff(y)[None,:]; W=U[4]*A; total=float(W.sum()); rho=U[0]; vx=U[1]/rho; vy=U[2]/rho; P=(s.GAMMA-1)*(U[3]-.5*(U[1]**2+U[2]**2)/rho); T=s.Tw*P/(s.P0_RAD*rho)
        assert np.isfinite(U).all() and rho.min()>0 and P.min()>0 and W.min()>=-1e-12
        if init is None: init=total;start={c:float(W[X<c].sum()) for c in (-40,0,40)}
        mx,my=[np.interp(t,ot,op[:,i]) for i in (0,1)]; ux,uy=[np.interp(t,ot,ov[:,i]) for i in (0,1)]; dx=X-mx;dy=Y-my; R=np.hypot(dx,dy);rr=np.sqrt(R**2+.5**2);phi=np.where(rr<=s.rt,np.interp(rr,r,pot),-s.G2*M/rr); B=.5*((vx-ux-s.OMEGA*dy)**2+(vy-uy+s.OMEGA*dx)**2)+s.GAMMA/(s.GAMMA-1)*P/rho+phi
        ii=np.argsort(T.ravel()); tc=T.ravel()[ii]; wc=np.cumsum(W.ravel()[ii]); q50=float(tc[np.searchsorted(wc,.5*total)]);q90=float(tc[np.searchsorted(wc,.9*total)])
        I=X<-40
        q=dict(dm=dm,t_Myr=t,mass=total,mass_drift=total/init-1,min_rho=float(rho.min()),min_p=float(P.min()),excess_tag=float((np.maximum(U[4]-rho,0)*A).sum()/total),centroid=float((W*X).sum()/total),J=float((W*(vy+2*s.OMEGA*X)).sum()/total),mean_T=float((W*T).sum()/total),T50=q50,T90=q90,fine_fraction=float(W[(np.diff(x)[:,None]<=1.00001)&(np.diff(y)[None,:]<=2.50001)].sum()/total),near40=float(W[R<40].sum()/total),near_rt=float(W[R<s.rt].sum()/total),Bneg40=float(W[(R<40)&(B<0)].sum()/total),inner_far40=float(W[I&(R>=40)].sum()/total),inner_far_rt=float(W[I&(R>=s.rt)].sum()/total),steps=int(a.step_count))
        for c in (-40,0,40):
            k='x'+str(c); inventory=float(W[X<c].sum()); q[f'inventory_{c}']=inventory/total; q[f'cum_in_{c}']=float(a['cum_in_'+k]); q[f'independent_budget_{c}']=(inventory-start[c])/init-(float(a['cum_in_'+k])+float(a['left_flux_'+k])-float(a['y_loss_'+k])); assert abs(q[f'independent_budget_{c}'])<1e-10
        assert abs(q['mean_T']-a.tag_mean_T_K)<1e-8 and abs(q['centroid']-a.tag_center_x_pc)<1e-9
        rows.append(q)
    p=d.merge(old,on='t_Myr',suffixes=('_wide','_original'))
    for _,a in p.iterrows():
        row=dict(dm=dm,t_Myr=float(a.t_Myr))
        for k in ('cum_in_x-40','cum_in_x0','cum_in_x40','tag_center_x_pc','tag_mean_T_K','delta_J'):
            row[k+'_wide']=float(a[k+'_wide']);row[k+'_original']=float(a[k+'_original']);row[k+'_diff']=float(a[k+'_wide']-a[k+'_original'])
        comparisons.append(row)
D=pd.DataFrame(rows).sort_values(['dm','t_Myr']); C=pd.DataFrame(comparisons); D.to_csv(root/'CP37_FIELDS.csv',index=False); C.to_csv(root/'CP37_ORIGINAL_COMPARISON.csv',index=False)
paired=[]
for t,g in D.groupby('t_Myr'):
    if len(g)!=2: continue
    d=g.set_index('dm');a=d.loc[1];c=d.loc[0];row={'t_Myr':float(t),**{k:float(a[k]) for k in ('centroid','J','mean_T','T50','T90','near40','near_rt','Bneg40','inner_far40','inner_far_rt','fine_fraction','inventory_-40')}}
    for cut in (-40,0,40): row[f'excess_{cut}']=float(a[f'cum_in_{cut}']-c[f'cum_in_{cut}']); row[f'control_inventory_{cut}']=float(c[f'inventory_{cut}'])
    paired.append(row)
P=pd.DataFrame(paired); P.to_csv(root/'CP37_PAIRED.csv',index=False); last=float(P.t_Myr.max()); end=P.iloc[-1].to_dict(); old=D.groupby('dm').t_Myr.max().to_dict()
summary={'last_by_dm':{str(k):float(v) for k,v in old.items()},'common_time':last,'full_pair_complete':bool(last==50),'fields':len(D),'max_budget':float(D.filter(regex='^independent_budget_').abs().max().max()),'max_mass_drift':float(D.mass_drift.abs().max()),'max_excess_tag':float(D.excess_tag.max()),'min_rho':float(D.min_rho.min()),'min_p':float(D.min_p.min()),'common_endpoint':end}
if last>=20:
    b=P[P.t_Myr==20].iloc[0];summary['late_excess_minus40']=float(end['excess_-40']-b['excess_-40'])
(root/'CP37_RESULTS.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':12,'axes.labelsize':12,'legend.fontsize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(9,6.7),layout='constrained'); fig.get_layout_engine().set(rect=(0,.11,1,1)); colors=['#0072B2','#D55E00','#009E73']
O=pd.read_csv(root/'original_DM1.csv'); Oc=pd.read_csv(root/'original_DM0.csv'); O=O.merge(Oc,on='t_Myr',suffixes=('_1','_0')); O=O[O.t_Myr<=last]
L=pd.read_csv(root/'CP35_PAIRED.csv');L=L[(L.mesh=='expanded')&(L.t_Myr<=last)].copy()
# For CP35 use a separate late crossing relative to the same t=20; no concatenation into independent histories.
for ax,k,ko,mult,lab in zip(axs.flat,['excess_-40','centroid','mean_T','fine_fraction'],[None,'tag_center_x_pc_1','tag_mean_T_K_1',None],[100,1,1,100],['Excess inward crossing (%)','Radial centroid (pc)','Mean tagged temperature (K)','Tag on fine cells (%)']):
    sets=[(P.t_Myr.to_numpy(),P[k].to_numpy()*mult,'Independent wide strip',colors[0])]
    if k=='excess_-40': sets.append((O.t_Myr.to_numpy(),(O['cum_in_x-40_1']-O['cum_in_x-40_0']).to_numpy()*100,'Original strip',colors[1]))
    elif ko: sets.append((O.t_Myr.to_numpy(),O[ko].to_numpy()*mult,'Original strip',colors[1]))
    if len(L)>1 and k in ('centroid','mean_T','fine_fraction'):
        q={'centroid':'x_centroid','mean_T':'mean_T','fine_fraction':'fine_fraction'}[k];sets.append((L.t_Myr.to_numpy(),L[q].to_numpy()*mult,'Late remapped strip',colors[2]))
    for t,v,l,col in sets:
        if len(t)>1:
            tt=np.linspace(t.min(),t.max(),2001);vv=PchipInterpolator(t,v)(tt);assert vv.min()>=v.min()-1e-8 and vv.max()<=v.max()+1e-8;ax.plot(tt,vv,lw=1.8,c=col,label=l)
        ax.scatter(t,v,s=10,c=col,zorder=3)
    ax.set(xlabel='Time (Myr)',ylabel=lab);ax.grid(alpha=.14)
handles={}
for ax in axs.flat:
    for h,l in zip(*ax.get_legend_handles_labels()):handles[l]=h
fig.legend(handles.values(),handles.keys(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=True);fig.savefig(root/'CP37_full_history_comparison.pdf');plt.close(fig)
