import numpy as np
d=np.load('/home/raulio/audit_20260714/estrategia/_data.npz',allow_pickle=True)
pm,pin,y,lh,la,ht,season=d['pm'],d['pin'],d['y'],d['lh'],d['la'],d['ht'],d['season']
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
print("=== DISPERSION DE LAS PROBABILIDADES ===")
for n,p in [("MODELO",pm),("PINNACLE",pin)]:
    print("  %-9s sd=%.4f  rango p1-p99=[%.3f, %.3f]  |p-0.5| medio=%.4f"
          %(n,p.std(),np.percentile(p,1),np.percentile(p,99),np.abs(p-0.5).mean()))
print("  correlacion logit(modelo) vs logit(pin) = %.4f"%np.corrcoef(logit(pm),logit(pin))[0,1])
print("  sd(logit modelo)=%.4f  sd(logit pin)=%.4f  ratio=%.2f"%(logit(pm).std(),logit(pin).std(),logit(pm).std()/logit(pin).std()))
print("  => el mercado separa los juegos %.1fx mas que nosotros"%(logit(pin).std()/logit(pm).std()))

def brier(p,yy): return float(np.mean((p-yy)**2))
def fit(X,yy,it=200):
    X=np.column_stack([np.ones(len(yy)),X]); b=np.zeros(X.shape[1])
    for _ in range(it):
        z=X@b; p=1/(1+np.exp(-z)); W=p*(1-p)+1e-12
        b=b+np.linalg.solve((X*W[:,None]).T@X+1e-8*np.eye(len(b)),X.T@(yy-p))
    return b
def pred(b,X): return 1/(1+np.exp(-(np.column_stack([np.ones(len(X)),X])@b)))

print("\n=== EL PIPELINE DE 9 MOTORES vs UNA LOGISTICA DE 2 PARAMETROS ===")
print("   (walk-forward por temporada: ajusto en 2024, evaluo en 2025, y al reves)")
r=np.log(lh/la)
ok=~np.isnan(r)
for tr,te in [(2024,2025),(2025,2024)]:
    A=(season==tr)&ok; B=(season==te)&ok
    b=fit(r[A][:,None],y[A]); p=pred(b,r[B][:,None])
    print("  ajuste %d -> test %d :  logistica sobre log(lh/la)  Brier=%.5f   pipeline completo=%.5f   Pinnacle=%.5f"
          %(tr,te,brier(p,y[B]),brier(pm[B],y[B]),brier(pin[B],y[B])))
print("  (la logistica usa las MISMAS lambdas que produce el pipeline; mide cuanto anade el Monte Carlo+Platt+sesgos)")

print("\n=== CUANTO DE PINNACLE ES REPRODUCIBLE CON NUESTRAS LAMBDAS ===")
for tr,te in [(2024,2025),(2025,2024)]:
    A=(season==tr)&ok; B=(season==te)&ok
    b=fit(r[A][:,None],logit(pin[A])*0+ (logit(pin[A])>0).astype(float))  # placeholder
    # regresion lineal logit(pin) ~ log(lh/la)
    X=np.column_stack([np.ones(A.sum()),r[A]]); coef=np.linalg.lstsq(X,logit(pin[A]),rcond=None)[0]
    Xb=np.column_stack([np.ones(B.sum()),r[B]]); yhat=Xb@coef
    ss=1-((logit(pin[B])-yhat)**2).sum()/((logit(pin[B])-logit(pin[B]).mean())**2).sum()
    print("  R2 fuera de muestra de log(lh/la) explicando logit(Pinnacle) [%d->%d]: %.3f"%(tr,te,ss))
X=np.column_stack([np.ones(ok.sum()),logit(pm[ok])])
coef=np.linalg.lstsq(X,logit(pin[ok]),rcond=None)[0]
res=logit(pin[ok])-X@coef
print("  R2 de logit(modelo) explicando logit(Pinnacle) (en muestra): %.3f"
      %(1-(res**2).sum()/((logit(pin[ok])-logit(pin[ok]).mean())**2).sum()))
print("  => %.0f%% de la varianza informativa del mercado NO esta en nuestra probabilidad"
      %(100*(res**2).sum()/((logit(pin[ok])-logit(pin[ok]).mean())**2).sum()))
