"""A2 - ROI real, significancia, y que pasa contra la MEJOR linea disponible."""
import sqlite3, numpy as np
d=np.load('/home/raulio/audit_20260714/estrategia/_data.npz',allow_pickle=True)
pm,pin,y,oh,oa,ht,gp,season=d['pm'],d['pin'],d['y'],d['oh'],d['oa'],d['ht'],d['gp'],d['season']
n=len(y)
over=1/oh+1/oa; vig=(over.mean()-1)/over.mean()*100
print("n=%d  breakeven/vig Pinnacle = %.3f%%"%(n,vig))

def roi_at(thresholds, price_h, price_a, tag):
    print("\n=== ROI 1u plano vs %s ==="%tag)
    for th in thresholds:
        # apostar el lado con mayor edge sobre el precio disponible
        ev_h=pm*price_h-1; ev_a=(1-pm)*price_a-1
        side=np.where(ev_h>=ev_a,1,0); ev=np.where(side==1,ev_h,ev_a)
        m=ev>=th
        if m.sum()==0: print("  edge>=%.0f%%  0 apuestas"%(th*100)); continue
        win=np.where(side==1,y,1-y)[m]
        price=np.where(side==1,price_h,price_a)[m]
        pnl=np.where(win==1,price-1,-1.0)
        roi=pnl.mean()*100
        # bootstrap agrupado por equipo local
        rng=np.random.default_rng(11); teams=np.unique(ht[m]); htm=ht[m]; bs=[]
        for _ in range(2000):
            pk=rng.choice(teams,len(teams),replace=True)
            idx=np.concatenate([np.where(htm==t)[0] for t in pk])
            bs.append(pnl[idx].mean()*100)
        lo,hi=np.percentile(bs,[2.5,97.5])
        print("  edge>=%4.0f%%  n=%4d  ROI=%+.2f%%  IC95 agrupado [%+.2f, %+.2f]  P(ROI>0)=%.1f%%"
              %(th*100,m.sum(),roi,lo,hi,100*(np.array(bs)>0).mean()))

roi_at([0,.02,.05,.08,.10], oh, oa, "PINNACLE")

# --- mejor linea disponible en el mercado, misma fecha ---
c=sqlite3.connect('/home/raulio/data/predictions_history.db')
rows=dict((r[0],r[1:]) for r in c.execute(
  "select game_pk, ml_home_best, ml_away_best, ml_home_best_bk, ml_away_best_bk, ml_home_cons, ml_away_cons, n_bookmakers from historical_odds"))
bh=np.array([rows.get(int(g),(np.nan,)*7)[0] or np.nan for g in gp],float)
ba=np.array([rows.get(int(g),(np.nan,)*7)[1] or np.nan for g in gp],float)
nb=np.array([rows.get(int(g),(np.nan,)*7)[6] or np.nan for g in gp],float)
cov=~np.isnan(bh)&~np.isnan(ba)
print("\ncobertura mejor-linea: %d/%d (%.1f%%)  mediana libros=%.0f"%(cov.sum(),n,100*cov.mean(),np.nanmedian(nb)))
if cov.sum()>0:
    ov_b=(1/bh+1/ba)
    print("overround de la MEJOR linea: media %.4f (vig %.3f%%)  vs Pinnacle %.4f"
          %(np.nanmean(ov_b),(np.nanmean(ov_b)-1)/np.nanmean(ov_b)*100,over.mean()))
    print("ventaja media de precio best-vs-pin: local %+.3f%%  visita %+.3f%%"
          %(100*np.nanmean(bh[cov]/oh[cov]-1),100*np.nanmean(ba[cov]/oa[cov]-1)))
    g=cov
    globals().update(dict(pm=pm[g],y=y[g],ht=ht[g]))
    roi_at([0,.02,.05,.08,.10], bh[g], ba[g], "MEJOR LINEA DISPONIBLE (n=%d)"%g.sum())
