"""A5 - Test definitivo de alfa: la componente de nuestra senal ortogonal al mercado, predice?"""
import numpy as np
d=np.load('/home/raulio/audit_20260714/estrategia/_data.npz',allow_pickle=True)
pm,pin,y,ht,season=d['pm'],d['pin'],d['y'],d['ht'],d['season']
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
Lm,Lp=logit(pm),logit(pin)
X=np.column_stack([np.ones(len(Lm)),Lp]); beta=np.linalg.lstsq(X,Lm,rcond=None)[0]
orth=Lm-X@beta   # la parte de nuestra vision que el mercado no tiene
print("descomposicion de nuestra senal: sd total=%.4f | paralela al mercado=%.4f | ORTOGONAL=%.4f"
      %(Lm.std(),(X@beta).std(),orth.std()))
print("=> %.0f%% de la varianza de nuestra probabilidad es informacion que el mercado NO refleja"%(100*orth.var()/Lm.var()))

def fit(Xf,yy,it=200):
    Xf=np.column_stack([np.ones(len(yy)),Xf]); b=np.zeros(Xf.shape[1])
    for _ in range(it):
        z=Xf@b; p=1/(1+np.exp(-z)); W=p*(1-p)+1e-12
        b=b+np.linalg.solve((Xf*W[:,None]).T@Xf+1e-8*np.eye(len(b)),Xf.T@(yy-p))
    return b
b=fit(np.column_stack([Lp,orth]),y)
print("\nlogistica  home_won ~ logit(pin) + ORTOGONAL:")
print("   b_pin=%+.4f   b_orth=%+.4f"%(b[1],b[2]))
rng=np.random.default_rng(3); teams=np.unique(ht); bs=[]
for _ in range(2000):
    pk=rng.choice(teams,len(teams),replace=True)
    idx=np.concatenate([np.where(ht==t)[0] for t in pk])
    try: bs.append(fit(np.column_stack([Lp[idx],orth[idx]]),y[idx])[2])
    except Exception: pass
bs=np.array(bs)
print("   IC95 agrupado por equipo de b_orth: [%+.4f, %+.4f]   P(b_orth>0)=%.1f%%"%(*np.percentile(bs,[2.5,97.5]),100*(bs>0).mean()))
print("   (si b_orth<=0, TODA nuestra desviacion del mercado es error, no alfa)")

print("\npor temporada (fuera de muestra cruzada):")
for tr,te in [(2024,2025),(2025,2024)]:
    A=season==tr; B=season==te
    bb=fit(np.column_stack([Lp[A],orth[A]]),y[A])
    print("   ajustado %d: b_orth=%+.4f   ->  aplicado a %d, Brier mezcla=%.5f vs pin sola=%.5f"
          %(tr,bb[2],te,np.mean((1/(1+np.exp(-(bb[0]+bb[1]*Lp[B]+bb[2]*orth[B])))-y[B])**2),np.mean((pin[B]-y[B])**2)))

print("\n=== cuanto alfa haria falta para ser rentable ===")
over=(1/d['oh']+1/d['oa']).mean()
print("  vig Pinnacle %.3f%% | vig mejor-linea 0.27%% | vig consenso 3.70%%"%((over-1)/over*100))
for target in [0.01,0.02,0.03]:
    print("  para +%.0f%% ROI plano contra Pinnacle hace falta una ventaja media de prob. de ~%.2fpp sobre el devig del mercado"
          %(target*100,100*(target+ (over-1)/over)/2.0))
