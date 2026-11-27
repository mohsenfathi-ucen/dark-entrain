from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
root=Path(__file__).resolve().parent
P=pd.read_csv(root/'CP37_PAIRED.csv'); last=P.t_Myr.max()
O=pd.read_csv(root/'original_DM1.csv').merge(pd.read_csv(root/'original_DM0.csv'),on='t_Myr',suffixes=('_1','_0'));O=O[O.t_Myr<=last]
L=pd.read_csv(root/'CP35_PAIRED.csv');L=L[L.t_Myr<=last];Lw=L[L.mesh=='expanded'];Lo=L[L.mesh=='original']
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':12,'axes.labelsize':12,'legend.fontsize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(9,6.9));fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.96,hspace=.37,wspace=.39)
colors=['#0072B2','#D55E00','#009E73']; labels=['Independent wide strip','Original strip','Late remapped strip']; styles=['-','--','-.'];handles={}
for ax,k,ko,mult,lab in zip(axs.flat,['excess_-40','centroid','mean_T','fine_fraction'],[None,'tag_center_x_pc_1','tag_mean_T_K_1',None],[100,1,1,100],['Excess inward crossing (%)','Radial centroid (pc)','Mean tagged temperature (K)','Tag on fine cells (%)']):
 sets=[(P.t_Myr,P[k]*mult,0)]
 if k=='excess_-40': sets.append((O.t_Myr,(O['cum_in_x-40_1']-O['cum_in_x-40_0'])*100,1))
 elif ko:sets.append((O.t_Myr,O[ko]*mult,1))
 elif k=='fine_fraction' and len(Lo):sets.append((Lo.t_Myr,Lo.fine_fraction*100,1))
 if k!='excess_-40' and len(Lw):sets.append((Lw.t_Myr,Lw[{'centroid':'x_centroid','mean_T':'mean_T','fine_fraction':'fine_fraction'}[k]]*mult,2))
 allv=[]
 for t,v,i in sets:
  t=np.asarray(t);v=np.asarray(v);assert np.isfinite(v).all();allv.extend(v.tolist())
  if len(t)>1:
   tt=np.linspace(t.min(),t.max(),2501);vv=PchipInterpolator(t,v)(tt);assert vv.min()>=v.min()-1e-8 and vv.max()<=v.max()+1e-8
   line,=ax.plot(tt,vv,lw=1.8,c=colors[i],ls=styles[i],label=labels[i]);handles[labels[i]]=line
  ax.scatter(t,v,s=11,c=colors[i],zorder=3)
 lo,hi=min(allv),max(allv); margin=max((hi-lo)*.055,1e-5);ax.set_ylim(lo-margin,hi+margin);ax.set_xlim(-53,max(-49,last)+3);assert all(ax.get_ylim()[0]<=v<=ax.get_ylim()[1] for v in allv);ax.set(xlabel='Time (Myr)',ylabel=lab);ax.grid(alpha=.14)
fig.legend([handles[k] for k in labels if k in handles],[k for k in labels if k in handles],loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=True);fig.savefig(root/'CP38_full_history_comparison.pdf');plt.close(fig)
