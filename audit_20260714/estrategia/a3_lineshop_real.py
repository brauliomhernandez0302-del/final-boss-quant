"""A3 - Cuanto de la ventaja de 'mejor linea' sobrevive a comisiones y a excluir exchanges."""
import sqlite3, numpy as np, json
d=np.load('/home/raulio/audit_20260714/estrategia/_data.npz',allow_pickle=True)
pm,pin,y,oh,oa,ht,gp=d['pm'],d['pin'],d['y'],d['oh'],d['oa'],d['ht'],d['gp']
c=sqlite3.connect('/home/raulio/data/predictions_history.db')
raw=c.execute("select game_pk, odds_api_id from historical_odds limit 1").fetchall()
# no guardamos el book-by-book completo; solo best/best_bk/cons/pin
H={int(r[0]):r[1:] for r in c.execute(
 "select game_pk,ml_home_best,ml_away_best,ml_home_best_bk,ml_away_best_bk,ml_home_cons,ml_away_cons from historical_odds")}
bh=np.array([H[int(g)][0] for g in gp],float); ba=np.array([H[int(g)][1] for g in gp],float)
bkh=np.array([H[int(g)][2] for g in gp]); bka=np.array([H[int(g)][3] for g in gp])
ch=np.array([H[int(g)][4] for g in gp],float); ca=np.array([H[int(g)][5] for g in gp],float)
EX={'Betfair','Matchbook','Smarkets','BetfairEX','Betfair Exchange'}
COMM={'Betfair':0.05,'Matchbook':0.02,'Smarkets':0.02}
print("share de mejores precios en exchange: local %.1f%%  visita %.1f%%"
      %(100*np.isin(bkh,list(EX)).mean(),100*np.isin(bka,list(EX)).mean()))
print("overround consenso (mediana libros): %.4f (vig %.3f%%)"%( (1/ch+1/ca).mean(), ((1/ch+1/ca).mean()-1)/(1/ch+1/ca).mean()*100))

def run(price_h,price_a,comm_h,comm_a,tag,ths=(0,.02,.05,.08,.10)):
    print("\n=== %s ==="%tag)
    ev_h=pm*price_h-1; ev_a=(1-pm)*price_a-1
    side=np.where(ev_h>=ev_a,1,0); ev=np.where(side==1,ev_h,ev_a)
    price=np.where(side==1,price_h,price_a); comm=np.where(side==1,comm_h,comm_a)
    win=np.where(side==1,y,1-y)
    pnl=np.where(win==1,(price-1)*(1-comm),-1.0)
    for th in ths:
        m=(ev>=th)&~np.isnan(price)
        if m.sum()<30: continue
        rng=np.random.default_rng(11); teams=np.unique(ht[m]); htm=ht[m]; p=pnl[m]; bs=[]
        for _ in range(1500):
            pk=rng.choice(teams,len(teams),replace=True)
            idx=np.concatenate([np.where(htm==t)[0] for t in pk]); bs.append(p[idx].mean()*100)
        lo,hi=np.percentile(bs,[2.5,97.5])
        print("  edge>=%4.0f%%  n=%4d  ROI=%+.2f%%  IC95 [%+.2f,%+.2f]  P(>0)=%.0f%%"%(th*100,m.sum(),p.mean()*100,lo,hi,100*(np.array(bs)>0).mean()))

zero=np.zeros(len(y))
run(bh,ba,zero,zero,"MEJOR LINEA, sin comision (optimista, irrealizable en exchange)")
comm_h=np.array([COMM.get(b,0.0) for b in bkh]); comm_a=np.array([COMM.get(b,0.0) for b in bka])
run(bh,ba,comm_h,comm_a,"MEJOR LINEA con comision de exchange (Betfair 5%, Matchbook 2%)")
# best entre libros tradicionales: no lo tenemos por libro -> cota inferior = consenso
run(ch,ca,zero,zero,"CONSENSO (mediana de libros) - proxy conservador sin exchanges")
run(oh,oa,zero,zero,"PINNACLE (referencia)")

# --- la senal de edge es anti-predictiva? ---
print("\n=== calibracion de la SENAL DE EDGE (edge = p_modelo - p_pin, lado local) ===")
edge=pm-pin
qs=np.quantile(edge,np.linspace(0,1,11))
print("  bucket        n   edge_medio  p_mod  p_pin  real   (real-p_mod)  (real-p_pin)")
for i in range(10):
    m=(edge>=qs[i])&(edge<=qs[i+1] if i==9 else edge<qs[i+1])
    print("  %2d  %5d  %+8.3f  %.4f %.4f %.4f  %+7.4f  %+7.4f"
          %(i+1,m.sum(),edge[m].mean(),pm[m].mean(),pin[m].mean(),y[m].mean(),y[m].mean()-pm[m].mean(),y[m].mean()-pin[m].mean()))
