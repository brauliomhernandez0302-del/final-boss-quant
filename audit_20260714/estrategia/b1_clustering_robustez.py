"""B1 - Sensibilidad de las conclusiones de A5/A2/A7 a la unidad de clustering
del bootstrap: equipo local (como usa el otro auditor), equipo visitante, fecha,
y par de equipos. Solo lectura, usa _data.npz ya generado por a1."""
import numpy as np
d = np.load('/home/raulio/audit_20260714/estrategia/_data.npz', allow_pickle=True)
pm, pin, y, ht, at, date, oh, oa = d['pm'], d['pin'], d['y'], d['ht'], d['at'], d['date'], d['oh'], d['oa']
n = len(y)
pair = np.array([f"{h}|{a}" for h, a in zip(ht, at)])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


Lm, Lp = logit(pm), logit(pin)
X = np.column_stack([np.ones(n), Lp])
beta = np.linalg.lstsq(X, Lm, rcond=None)[0]
orth = Lm - X @ beta


def fit(Xf, yy, it=200):
    Xf = np.column_stack([np.ones(len(yy)), Xf])
    b = np.zeros(Xf.shape[1])
    for _ in range(it):
        z = Xf @ b
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-12
        b = b + np.linalg.solve((Xf * W[:, None]).T @ Xf + 1e-8 * np.eye(len(b)), Xf.T @ (yy - p))
    return b


def cluster_boot_orth(cluster_ids, B=1500, seed=3):
    rng = np.random.default_rng(seed)
    idxmap = {c: np.where(cluster_ids == c)[0] for c in np.unique(cluster_ids)}
    clusters = list(idxmap)
    bs = []
    for _ in range(B):
        pick = rng.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([idxmap[c] for c in pick])
        try:
            bs.append(fit(np.column_stack([Lp[idx], orth[idx]]), y[idx])[2])
        except Exception:
            pass
    bs = np.array(bs)
    return np.percentile(bs, [2.5, 97.5]), 100 * (bs > 0).mean(), len(clusters)


print("=== b_orth (A5): 'nuestra desviacion del mercado, predice?' bajo distintos clusters ===")
for name, ids in [("equipo LOCAL (el que usa el otro auditor)", ht),
                   ("equipo VISITANTE", at),
                   ("fecha (363 dias)", date),
                   ("par local|visita (856 pares)", pair)]:
    (lo, hi), pgt0, ncl = cluster_boot_orth(ids)
    print(f"  cluster={name:42s} n_clusters={ncl:4d}  IC95 b_orth=[{lo:+.4f},{hi:+.4f}]  P(>0)={pgt0:5.1f}%")

print("\n=== ROI vs Pinnacle, edge>=0, bajo distintos clusters ===")
ev_h = pm * oh - 1
ev_a = (1 - pm) * oa - 1
side = np.where(ev_h >= ev_a, 1, 0)
ev = np.where(side == 1, ev_h, ev_a)
win = np.where(side == 1, y, 1 - y)
price = np.where(side == 1, oh, oa)
pnl = np.where(win == 1, price - 1, -1.0)
m = ev >= 0


def cluster_boot_roi(cluster_ids, mask, B=2000, seed=11):
    rng = np.random.default_rng(seed)
    cids = cluster_ids[mask]
    p_ = pnl[mask]
    idxmap = {c: np.where(cids == c)[0] for c in np.unique(cids)}
    clusters = list(idxmap)
    bs = []
    for _ in range(B):
        pick = rng.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([idxmap[c] for c in pick])
        bs.append(p_[idx].mean() * 100)
    bs = np.array(bs)
    return np.percentile(bs, [2.5, 97.5]), 100 * (bs > 0).mean(), len(clusters)


for name, ids in [("equipo LOCAL", ht), ("equipo VISITANTE", at),
                   ("fecha", date), ("par local|visita", pair)]:
    (lo, hi), pgt0, ncl = cluster_boot_roi(ids, m)
    print(f"  cluster={name:20s} n_clusters={ncl:4d}  ROI={pnl[m].mean()*100:+.2f}%  IC95=[{lo:+.2f},{hi:+.2f}]  P(>0)={pgt0:5.1f}%")

print("\n=== dos-vias (cluster por equipo local Y visitante simultaneamente, Cameron-Gelbach-Miller) ===")
# CGM: SE^2 = SE_local^2 + SE_visita^2 - SE_interseccion^2(par)
def boot_var(cluster_ids, mask, target='orth', B=1500, seed=21):
    rng = np.random.default_rng(seed)
    if target == 'orth':
        cids = cluster_ids
        idxmap = {c: np.where(cids == c)[0] for c in np.unique(cids)}
        clusters = list(idxmap)
        bs = []
        for _ in range(B):
            pick = rng.choice(clusters, len(clusters), replace=True)
            idx = np.concatenate([idxmap[c] for c in pick])
            try:
                bs.append(fit(np.column_stack([Lp[idx], orth[idx]]), y[idx])[2])
            except Exception:
                pass
        return np.var(bs)

v_home = boot_var(ht, None, 'orth')
v_away = boot_var(at, None, 'orth')
v_pair = boot_var(pair, None, 'orth')
v_cgm = v_home + v_away - v_pair
se_cgm = np.sqrt(max(v_cgm, v_pair))  # piso conservador si la resta da negativo
b_point = fit(np.column_stack([Lp, orth]), y)[2]
print(f"  b_orth punto={b_point:+.4f}  SE(local)={np.sqrt(v_home):.4f}  SE(visita)={np.sqrt(v_away):.4f}  SE(par)={np.sqrt(v_pair):.4f}")
print(f"  SE CGM (2-vias) = {se_cgm:.4f}  ->  IC95 aprox [{b_point-1.96*se_cgm:+.4f}, {b_point+1.96*se_cgm:+.4f}]")
