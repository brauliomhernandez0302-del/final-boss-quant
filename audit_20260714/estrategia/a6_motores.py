"""A6 - Que motor mueve lambda, cuanto, y si su movimiento coincide con el mercado."""
import sqlite3,json,numpy as np
from collections import defaultdict
c=sqlite3.connect('/home/raulio/data/predictions_history.db')
rows=c.execute("""select game_pk,backtest_stage_factors_json,ml_home_pin,ml_away_pin,home_won
 from game_outcomes where source='backtest' and season in (2024,2025)
 and substr(backtest_run_at,1,10)='2026-07-30' and backtest_stage_factors_json is not null""").fetchall()
print("juegos con stage_factors:",len(rows))
sf=defaultdict(list); gp=[]; pinl=[]; yl=[]
keys=set()
for g,j,oh,oa,y in rows:
    d=json.loads(j); keys|=set(d.keys())
for g,j,oh,oa,y in rows:
    d=json.loads(j)
    if not oh or not oa: continue
    for k in keys: sf[k].append(float(d.get(k,1.0)) if isinstance(d.get(k,1.0),(int,float)) else np.nan)
    ih,ia=1/oh,1/oa; pinl.append(ih/(ih+ia)); yl.append(y)
pin=np.array(pinl,float); y=np.array(yl,float)
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
Lp=logit(pin)
print("\n%-34s %7s %8s %8s %8s %9s"%("stage factor","activo%","p10","p90","sd","corr c/ mercado*"))
res=[]
for k in sorted(keys):
    v=np.array(sf[k],float)
    if np.all(np.isnan(v)): continue
    act=100*np.mean(np.abs(v-1.0)>1e-6)
    if act<1: continue
    lv=np.log(np.clip(v,1e-6,None))
    ok=~np.isnan(lv)&np.isfinite(lv)
    cr=np.corrcoef(lv[ok],Lp[ok])[0,1] if lv[ok].std()>1e-9 else np.nan
    res.append((act,k,np.nanpercentile(v,10),np.nanpercentile(v,90),np.nanstd(lv),cr))
for act,k,p10,p90,sd,cr in sorted(res,key=lambda r:-r[4]):
    print("%-34s %6.1f%% %8.4f %8.4f %8.4f %9s"%(k,act,p10,p90,sd,("%+.3f"%cr) if cr==cr else "n/a"))
print("\n*corr entre log(factor) y logit(prob Pinnacle del local). Factores del lado LOCAL deberian correlacionar +;")
print(" los del visitante -. Cerca de 0 = ese motor mueve lambda en una direccion que el mercado no reconoce.")
