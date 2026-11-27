"""WS4.1 magnetic-moment label analysis and descriptor baselines on IMP2D.

Reads the raw IMP2D ASE database, applies the source filters (converged,
finite eform, |eform| <= 20 eV; 10 641 rows), excludes the intrinsically
magnetic host Cr2I6, and evaluates |spin| (integer total supercell moment,
capped at 8 muB) with grouped 5-fold CV under pair / host / dopant holdout.

This is a quick learnability check, not the audited 10 224 set. Rerun on the
audited set once WS0.1 lands.

Usage:
    python scripts/prm/ws4_magmom_baseline.py --db data/raw/imp2d.db \
        --out results/prm/ws4_magmom_baseline.json
"""
import argparse, json, math
import numpy as np, pandas as pd
from ase.db import connect
from ase.data import atomic_numbers
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/raw/imp2d.db")
ap.add_argument("--out", default="results/prm/ws4_magmom_baseline.json")
args = ap.parse_args()

rec = []
for row in connect(args.db).select():
    e = row.eform
    if not row.converged or e is None or not math.isfinite(e) or abs(e) > 20:
        continue
    Z = row.numbers
    rec.append(dict(host=row.host, dopant=row.dopant, site=row.site,
                    defecttype=row.defecttype, spin=int(row.spin), natoms=len(Z),
                    sumZ=int(Z.sum()), depth=row.depth, ext=row.extension_factor))
df_all = pd.DataFrame(rec)
df_all["m_raw"] = df_all.spin.abs()
parity_all = float(((df_all.m_raw % 2) == (df_all.sumZ % 2)).mean())
df = df_all[df_all.host != "Cr2I6"].reset_index(drop=True)   # intrinsically magnetic host
df["m"] = df.m_raw.clip(upper=8)

# free-atom Hund's-rule moment via aufbau (Madelung order + common exceptions)
order = ['1s','2s','2p','3s','3p','4s','3d','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p']
cap = {'s':2,'p':6,'d':10,'f':14}
exc = {24:{'4s':1,'3d':5},29:{'4s':1,'3d':10},41:{'5s':1,'4d':4},42:{'5s':1,'4d':5},44:{'5s':1,'4d':7},
       45:{'5s':1,'4d':8},46:{'5s':0,'4d':10},47:{'5s':1,'4d':10},78:{'6s':1,'5d':9},79:{'6s':1,'5d':10},
       57:{'4f':0,'5d':1},64:{'4f':7,'5d':1}}
def config(Z):
    left=Z; c={}
    for o in order:
        n=min(cap[o[-1]],left); c[o]=n; left-=n
        if left==0: break
    if Z in exc:
        for k,v in exc[Z].items(): c[k]=v
        # rebalance: remaining electrons sit in the highest open shell already set
        diff = Z-sum(c.values())
        if diff: c[[k for k in exc[Z]][-1]] += diff
    return c
def hund(Z):
    s=0
    for o,n in config(Z).items():
        L=cap[o[-1]]//2; s += n if n<=L else 2*L-n
    return s
def group_period(Z):
    from ase.data import chemical_symbols
    periods=[2,10,18,36,54,86,118]; p=next(i+1 for i,b in enumerate(periods) if Z<=b)
    start=[0,2,10,18,36,54,86][p-1]; k=Z-start
    if p<=3: g = k if k<=2 else k+10
    elif p<=5: g=k
    else: g = k if k<=2 else (3 if k<=17 else k-14)
    return g,p
df['Zd']=df.dopant.map(atomic_numbers)
df['hund']=df.Zd.map(hund)
gp=df.Zd.map(group_period); df['grp']=[g for g,p in gp]; df['per']=[p for g,p in gp]
df['parity']=df.sumZ%2
df['is_int']=(df.defecttype=='interstitial').astype(int)
df['site_i']=df.site.str.extract(r'(\d+)').astype(int)
hosts=sorted(df.host.unique()); df['host_i']=df.host.map({h:i for i,h in enumerate(hosts)})
df['pair']=df.host+'_'+df.dopant

def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    mag_y=y>0; mag_p=p>0
    bacc=0.5*((mag_p[mag_y]).mean()+(~mag_p[~mag_y]).mean())
    return dict(exact=(y==p).mean(), within1=(abs(y-p)<=1).mean(), mae=abs(y-p).mean(), bacc_mag=bacc)

feats_pair=['Zd','hund','grp','per','parity','is_int','site_i','host_i','natoms','depth','ext']
feats_host=['Zd','hund','grp','per','parity','is_int','site_i','natoms','depth','ext']
out={}
for name,groups,feats in [('pair-holdout',df.pair,feats_pair),('host-holdout',df.host,feats_host),('dopant-holdout',df.dopant,feats_pair)]:
    pred=np.zeros(len(df),int); base_mode=np.zeros(len(df),int)
    for tr,te in GroupKFold(5).split(df,groups=groups):
        clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,categorical_features=[feats.index('host_i')] if 'host_i' in feats else None,random_state=0)
        clf.fit(df.loc[tr,feats],df.m[tr]); pred[te]=clf.predict(df.loc[te,feats])
        # lookup baseline: mode moment of the same dopant (pair/host holdout) in training rows, else parity
        mode=df.loc[tr].groupby('dopant').m.agg(lambda s:s.mode().iloc[0])
        base_mode[te]=df.loc[te,'dopant'].map(mode).fillna(df.loc[te,'parity']).astype(int)
    out[name]={'GBM':metrics(df.m,pred),'dopant-mode lookup':metrics(df.m,base_mode)}
parity_only = df.parity.values
hund_b = df.hund.clip(upper=8).values
clean = lambda d: {a: round(float(x), 4) for a, x in d.items()}
res = {
    "n_filtered": int(len(df_all)), "n_used": int(len(df)), "excluded_hosts": ["Cr2I6"],
    "n_hosts": int(df.host.nunique()),
    "frac_nonmagnetic": round(float((df.m == 0).mean()), 4),
    "spin_is_integer": True,
    "parity_rule_holds_frac_all": round(parity_all, 4),
    "baselines": {"parity_only": clean(metrics(df.m, parity_only)),
                  "free_atom_hund": clean(metrics(df.m, hund_b))},
    "grouped_cv": {k: {b: clean(m) for b, m in v.items()} for k, v in out.items()},
}
print(json.dumps(res, indent=1))
import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
