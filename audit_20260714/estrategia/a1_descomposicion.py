"""A1 - Descomposicion de la brecha modelo-vs-Pinnacle. Solo lectura."""
import sqlite3, numpy as np, json, math

DB='/home/raulio/data/predictions_history.db'
c=sqlite3.connect(DB)
rows=c.execute("""
 select game_pk, official_date, season, home_team, away_team,
        backtest_p_home, ml_home_pin, ml_away_pin, home_won,
        actual_home_runs, actual_away_runs, backtest_lambda_home, backtest_lambda_away
 from game_outcomes
 where source='backtest' and season in (2024,2025)
   and backtest_p_home is not null and home_won is not null
   and substr(backtest_run_at,1,10)='2026-07-30'
""").fetchall()
print("juegos corrida canonica:", len(rows))

gp=np.array([r[0] for r in rows]); date=np.array([r[1] for r in rows])
season=np.array([r[2] for r in rows]); ht=np.array([r[3] for r in rows]); at=np.array([r[4] for r in rows])
pm=np.array([r[5] for r in rows],float)
oh=np.array([r[6] if r[6] else np.nan for r in rows],float)
oa=np.array([r[7] if r[7] else np.nan for r in rows],float)
y=np.array([r[8] for r in rows],float)
rh=np.array([r[9] if r[9] is not None else np.nan for r in rows],float)
ra=np.array([r[10] if r[10] is not None else np.nan for r in rows],float)
lh=np.array([r[11] if r[11] is not None else np.nan for r in rows],float)
la=np.array([r[12] if r[12] is not None else np.nan for r in rows],float)

has=~np.isnan(oh)&~np.isnan(oa)
print("con Pinnacle:", has.sum())
ih,ia=1/oh,1/oa
over=ih+ia
pin=ih/over   # devig proporcional
print("overround Pinnacle: media %.4f  mediana %.4f  p90 %.4f"%(np.nanmean(over),np.nanmedian(over),np.nanpercentile(over[has],90)))
print("vig por lado (breakeven que hay que superar): %.3f%%"%((np.nanmean(over)-1)/np.nanmean(over)*100))

S=has
pm_,pin_,y_=pm[S],pin[S],y[S]
def brier(p,y): return float(np.mean((p-y)**2))
def ll(p,y):
    p=np.clip(p,1e-9,1-1e-9); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
base=float(y_.mean())
print("\n--- ESCALERA (n=%d, tasa local %.4f) ---"%(S.sum(),base))
print("moneda 0.5          %.5f"%brier(np.full_like(y_,0.5),y_))
print("constante=tasa      %.5f"%brier(np.full_like(y_,base),y_))
print("MODELO              %.5f"%brier(pm_,y_))
print("PINNACLE            %.5f"%brier(pin_,y_))
print("brecha modelo-pin   %.5f"%(brier(pm_,y_)-brier(pin_,y_)))

# --- Murphy: reliability / resolution / uncertainty ---
def murphy(p,y,nb=12):
    edges=np.quantile(p,np.linspace(0,1,nb+1)); edges[0]-=1e-9; edges[-1]+=1e-9
    idx=np.digitize(p,edges)-1; idx=np.clip(idx,0,nb-1)
    n=len(y); ybar=y.mean(); rel=res=0.0; det=[]
    for k in range(nb):
        m=idx==k
        if m.sum()==0: continue
        nk=m.sum(); pk=p[m].mean(); ok=y[m].mean()
        rel+=nk*(pk-ok)**2; res+=nk*(ok-ybar)**2
        det.append((nk,round(pk,4),round(ok,4)))
    return rel/n,res/n,ybar*(1-ybar),det
for name,p in [("MODELO",pm_),("PINNACLE",pin_)]:
    rel,res,unc,det=murphy(p,y_)
    print("\n%s  reliability=%.5f  resolution=%.5f  uncertainty=%.5f  ->  B=%.5f"%(name,rel,res,unc,unc-res+rel))
    for d in det: print("   n=%4d  pred=%.4f  real=%.4f"%d)

# --- Regresion logistica: aporta el modelo algo sobre Pinnacle? ---
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def fit(X,y,iters=300):
    X=np.column_stack([np.ones(len(y)),X]); b=np.zeros(X.shape[1])
    for _ in range(iters):
        z=X@b; p=1/(1+np.exp(-z)); W=p*(1-p)+1e-12
        g=X.T@(y-p); H=(X*W[:,None]).T@X
        b=b+np.linalg.solve(H+1e-8*np.eye(len(b)),g)
    return b
Lm,Lp=logit(pm_),logit(pin_)
b_p=fit(Lp[:,None],y_);           print("\nsolo Pinnacle:  a=%.4f b_pin=%.4f"%tuple(b_p))
b_m=fit(Lm[:,None],y_);           print("solo modelo:    a=%.4f b_mod=%.4f"%tuple(b_m))
b_b=fit(np.column_stack([Lp,Lm]),y_); print("ambos:          a=%.4f b_pin=%.4f b_mod=%.4f"%tuple(b_b))

# bootstrap agrupado por equipo local (los equipos se repiten)
teams=np.unique(ht[S]); rng=np.random.default_rng(7); coefs=[]
htS=ht[S]
for _ in range(400):
    pick=rng.choice(teams,len(teams),replace=True)
    idx=np.concatenate([np.where(htS==t)[0] for t in pick])
    try: coefs.append(fit(np.column_stack([Lp[idx],Lm[idx]]),y_[idx]))
    except Exception: pass
coefs=np.array(coefs)
print("bootstrap agrupado por equipo local (n=%d):"%len(coefs))
print("  b_pin  IC95 [%.4f, %.4f]"%tuple(np.percentile(coefs[:,1],[2.5,97.5])))
print("  b_mod  IC95 [%.4f, %.4f]   P(b_mod>0)=%.1f%%"%(*np.percentile(coefs[:,2],[2.5,97.5]),100*(coefs[:,2]>0).mean()))

# --- Cuanta brecha cierra la recalibracion sola (cota superior, en-muestra) ---
pm_cal=1/(1+np.exp(-(b_m[0]+b_m[1]*Lm)))
print("\nmodelo recalibrado (Platt en-muestra, COTA SUPERIOR optimista): %.5f  (cierra %.1f%% de la brecha)"
      %(brier(pm_cal,y_), 100*(brier(pm_,y_)-brier(pm_cal,y_))/(brier(pm_,y_)-brier(pin_,y_))))
# isotonica en-muestra
o=np.argsort(pm_); ys=y_[o]
def pava(v):
    v=v.astype(float).copy(); w=np.ones(len(v)); i=0
    lvl=list(v); wt=list(w); 
    out=[]; 
    for val,ww in zip(lvl,wt):
        out.append([val,ww])
        while len(out)>1 and out[-2][0]>out[-1][0]:
            v2,w2=out.pop(); v1,w1=out.pop()
            out.append([(v1*w1+v2*w2)/(w1+w2),w1+w2])
    r=[]
    for val,ww in out: r+= [val]*int(ww)
    return np.array(r)
iso=np.empty(len(y_)); iso[o]=pava(ys)
print("modelo isotonico en-muestra (COTA SUPERIOR muy optimista):     %.5f  (cierra %.1f%%)"
      %(brier(iso,y_),100*(brier(pm_,y_)-brier(iso,y_))/(brier(pm_,y_)-brier(pin_,y_))))

# --- Mezcla optima modelo+mercado ---
print("\n--- mezcla en logit: p = sigmoid((1-w)*logit(pin) + w*logit(modelo)), reescalada ---")
best=None
for w in np.arange(0,1.01,0.05):
    z=(1-w)*Lp+w*Lm; p=1/(1+np.exp(-z)); b=brier(p,y_)
    if best is None or b<best[1]: best=(w,b)
    if abs(w*20-round(w*20))<1e-9 and round(w*10)==w*10: print("  w=%.2f  Brier=%.5f"%(w,b))
print("  optimo w=%.2f  Brier=%.5f"%best)

np.savez('/home/raulio/audit_20260714/estrategia/_data.npz',
         gp=gp[S],date=date[S],season=season[S],ht=htS,at=at[S],pm=pm_,pin=pin_,y=y_,
         oh=oh[S],oa=oa[S],rh=rh[S],ra=ra[S],lh=lh[S],la=la[S])
