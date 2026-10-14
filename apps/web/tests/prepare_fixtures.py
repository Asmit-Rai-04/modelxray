"""Generate test evidence with the unmodified backend, never hand-made analytics."""
import os
import sys
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

web = Path(__file__).resolve().parents[1]
out = web / '.test-artifacts'
out.mkdir(exist_ok=True)
os.chdir(out)
sys.path.insert(0, str(web.parents[1] / 'apps' / 'api'))
from modelxray.main import investigate_demo, get_investigation, compare_demo, DemoInvestigationRequest, job_manager

request = DemoInvestigationRequest(rows=2500, budget=24, random_state=42)
# Exercise the persisted response used by async jobs and recovery, including
# the backend-assigned counterexample identifiers and their region links.
result = get_investigation(investigate_demo(request)['investigation_id'])
(out / 'investigation.json').write_text(json.dumps(result), encoding='utf-8')
(out / 'comparison.json').write_text(json.dumps(compare_demo(request)), encoding='utf-8')
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=7)
frame = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
for version, depth in [(1, 2), (2, 6)]:
    model = RandomForestClassifier(n_estimators=20, max_depth=depth, random_state=version).fit(frame.iloc[:500], y[:500])
    joblib.dump(model, out / f'model-v{version}.joblib')
evaluation = frame.iloc[500:].copy()
evaluation['target'] = y[500:]
evaluation.to_csv(out / 'evaluation.csv', index=False)
job_manager.shutdown()
print(f'Generated real backend evidence: {len(result["failure_atlas"])} regions, {result["active_investigation"]["experiments_executed"]} experiments')
