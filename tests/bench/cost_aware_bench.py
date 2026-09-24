"""Compare cost-unaware vs cost-aware experiment allocation on identical seeds."""
from __future__ import annotations

import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from modelxray.investigation.controller import run_active_investigation
from tests.bench.scenarios import SCENARIOS, CONTROL_SCENARIOS

TRAIN_FRACTION = .55
DISCOVERY_FRACTION = .225

class CountingAdapter:
    def __init__(self, model): self.model=model; self.calls=0; self.rows=0
    def predict(self, X): self.calls += 1; self.rows += len(X); return np.asarray(self.model.predict(X))
    @property
    def classes_(self): return np.asarray(getattr(self.model, 'classes_', []))

def _mask(condition: str, X: pd.DataFrame):
    import re
    if ' ∈ [' in condition:
        terms, bounds = condition.rsplit(' ∈ [', 1); lo, hi = bounds.rstrip(']').split(',', 1)
        score=np.zeros(len(X)); valid=np.ones(len(X), dtype=bool)
        for wt, feat in re.findall(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\*([A-Za-z_][A-Za-z0-9_]*)', terms):
            v=pd.to_numeric(X[feat], errors='coerce').to_numpy(float); valid &= np.isfinite(v); score += float(wt)*v
        return valid & (score >= float(lo)) & (score <= float(hi))
    m=np.ones(len(X), dtype=bool)
    for part in condition.split(' & '):
        f, op, thr = part.strip().rsplit(' ', 2); v=pd.to_numeric(X[f], errors='coerce').to_numpy(float)
        m &= np.isfinite(v) & ((v <= float(thr)) if op == '<=' else (v > float(thr)))
    return m

def one(scenario, seed, budget, cost_aware):
    frame, y, truth = SCENARIOS[scenario](seed=seed)
    idx=np.arange(len(frame)); tr, ev=train_test_split(idx, test_size=1-TRAIN_FRACTION, random_state=seed)
    disc_n=int(len(ev)*DISCOVERY_FRACTION/(1-TRAIN_FRACTION)); di, va=ev[:disc_n], ev[disc_n:]
    if scenario in CONTROL_SCENARIOS:
        model=RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(frame.iloc[tr], y[tr])
    else:
        model=LogisticRegression(max_iter=1000).fit(frame.iloc[tr], y[tr])
    adapter=CountingAdapter(model); t=time.perf_counter()
    run=run_active_investigation(adapter, frame.iloc[di].reset_index(drop=True), pd.Series(y[di]), frame.iloc[va].reset_index(drop=True), pd.Series(y[va]), budget=budget, cost_aware=cost_aware)
    runtime=time.perf_counter()-t
    promoted=[o for o in run.observations if o['reproducible']]
    best=0.0
    if truth is not None:
        Xev=frame.iloc[ev].reset_index(drop=True); truth_ev=np.asarray(truth[ev], dtype=bool)
        for o in promoted:
            m=_mask(o['condition'], Xev); tp=np.logical_and(m, truth_ev).sum(); fp=np.logical_and(m, ~truth_ev).sum(); fn=np.logical_and(~m, truth_ev).sum()
            if tp:
                p=tp/(tp+fp); r=tp/(tp+fn); best=max(best, float(2*p*r/(p+r)) if p+r else 0.0)
    return {'discovered': bool(best >= .5), 'f1': best, 'predict_calls': int(adapter.calls), 'runtime_s': runtime, 'promoted': len(promoted)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--budget', type=int, default=24); ap.add_argument('--out', default='benchmark/cost_aware_compare.json'); ap.add_argument('--workers', type=int, default=1); args=ap.parse_args()
    scenarios=['S1_main_effect','S2_interaction','S3_boundary_instability','S4_covariate_shift','S5_noisy_feature','S6_correlated_features','S7_class_imbalance']
    data={}
    tasks = [(s, seed, args.budget, mode) for mode in (False, True) for s in scenarios for seed in range(100, 100 + args.seeds)]
    rows_by_mode = {False: [], True: []}
    if args.workers <= 1:
        for s, seed, budget, mode in tasks:
            rows_by_mode[mode].append({'scenario': s, 'seed': seed, **one(s, seed, budget, mode)})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, s, seed, budget, mode): (s, seed, mode) for s, seed, budget, mode in tasks}
            for fut in as_completed(futures):
                s, seed, mode = futures[fut]
                rows_by_mode[mode].append({'scenario': s, 'seed': seed, **fut.result()})
    for mode in (False, True):
        data['cost_aware' if mode else 'baseline'] = sorted(rows_by_mode[mode], key=lambda r: (r['scenario'], r['seed']))
    summary={}
    for mode,rows in data.items():
        summary[mode]={}
        for s in scenarios:
            rr=[r for r in rows if r['scenario']==s]
            summary[mode][s]={
                'discovery_rate': float(np.mean([r['discovered'] for r in rr])),
                'mean_f1': float(np.mean([r['f1'] for r in rr])),
                'mean_predict_calls': float(np.mean([r['predict_calls'] for r in rr])),
                'mean_runtime_s': float(np.mean([r['runtime_s'] for r in rr])),
                'mean_promoted': float(np.mean([r['promoted'] for r in rr])),
            }
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({'config':vars(args),'summary':summary,'rows':data},indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
