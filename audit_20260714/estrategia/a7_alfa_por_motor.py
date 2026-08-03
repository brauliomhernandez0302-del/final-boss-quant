"""A7 - Que componente predice el resultado MAS ALLA del mercado. Solo lectura."""
import sqlite3,json,numpy as np
c=sqlite3.connect('/home/raulio/data/predictions_history.db')
rows=c.execute("""select game_pk,home_team,backtest_stage_factors_json,ml_home_pin,ml_away_pin,home_won,season
 from game_outcomes where source='backtest' and season in (2024,2025)
 and substr(backtest_run_at,1,10)='2026-07-30' and ml_home_pin is not null and backtest_stage_factors_json is not null""").fetchall()
d=[json.loads(r[2]) for r in rows]
ht=np.array([r[1] for r in rows]); y=np.array([r[5] for r in rows],float); season=np.array([r[6] for r in rows])
oh=np.array([r[3] for r in rows],float); oa=np.array([r[4] for r in rows],float)
pin=(1/oh)/((1/oh)+(1/oa)); Lp=np.log(pin/(1-pin)); n=len(rows)
def gv(k,i,dv=1.0):
    v=d[i].get(k,dv); return float(v) if isinstance(v,(int,float)) else dv
SPEC={'l0(TTE ofensa)':('l0_home_lambda','l0_away_lambda'),
      'pitcher':('pitcher_on_home_lambda','pitcher_on_away_lambda'),
      'bias(equipo)':('bias_on_home_lambda','bias_on_away_lambda'),
      'bullpen':('bullpen_on_home_lambda','bullpen_on_away_lambda'),
      'defense':('defense_on_home_lambda','defense_on_away_lambda'),
      'park':('park_on_home_lambda','park_on_away_lambda'),
      'context':('context_on_home_lambda','context_on_away_lambda'),
      'hfa':('hfa_on_home_lambda','hfa_on_away_lambda')}
comp={}
for name,(kh,ka) in SPEC.items():
    h=np.array([gv(kh,i) for i in range(n)]); a=np.array([gv(ka,i) for i in range(n)])
    x=np.log(np.clip(h,1e-9,None))-np.log(np.clip(a,1e-9,None))
    comp[name]=(x-x.mean())/x.std() if x.std()>1e-9 else None
def fit(X,yy,it=30):
    X=np.column_stack([np.ones(len(yy)),X]); b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-(X@b))); W=p*(1-p)+1e-12
        b=b+np.linalg.solve((X*W[:,None]).T@X+1e-8*np.eye(len(b)),X.T@(yy-p))
    return b
idxmap={t:np.where(ht==t)[0] for t in np.unique(ht)}; teams=list(idxmap)
rng=np.random.default_rng(5)
BOOT=[np.concatenate([idxmap[t] for t in rng.choice(teams,len(teams),replace=True)]) for _ in range(800)]
print("home_won ~ logit(Pinnacle) + motor    (motor = log(factor_local/factor_visita), estandarizado)")
print("coef>0 con IC que no cruza 0 = ese motor aporta informacion que el mercado NO tiene\n")
print("%-16s %9s %24s %8s   %s"%("motor","coef","IC95 agrupado x equipo","P(>0)","coef 2024 / 2025"))
for k,x in comp.items():
    if x is None: print("%-16s   (mueve identico ambos lados: aporte al moneyline = CERO por construccion)"%k); continue
    b=fit(np.column_stack([Lp,x]),y)
    bs=np.array([fit(np.column_stack([Lp[i],x[i]]),y[i])[2] for i in BOOT])
    lo,hi=np.percentile(bs,[2.5,97.5])
    c24=fit(np.column_stack([Lp[season==2024],x[season==2024]]),y[season==2024])[2]
    c25=fit(np.column_stack([Lp[season==2025],x[season==2025]]),y[season==2025])[2]
    print("%-16s %+9.4f    [%+7.4f, %+7.4f] %7.1f%%   %+.4f / %+.4f"%(k,b[2],lo,hi,100*(bs>0).mean(),c24,c25))
act=[k for k,v in comp.items() if v is not None]
X=np.column_stack([Lp]+[comp[k] for k in act]); b=fit(X,y)
print("\ntodos juntos: b_pin=%+.4f | "%b[1]+" ".join("%s=%+.4f"%(k,b[2+i]) for i,k in enumerate(act)))
X2=np.column_stack([comp[k] for k in act]); b2=fit(X2,y)
p2=1/(1+np.exp(-(np.column_stack([np.ones(n)]+[comp[k] for k in act])@b2)))
print("control solo-motores (sin mercado, EN MUESTRA): Brier=%.5f | constante=%.5f | Pinnacle=%.5f"
      %(np.mean((p2-y)**2),np.mean((y.mean()-y)**2),np.mean((pin-y)**2)))
