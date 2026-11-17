"""Independent reconciliation against source predictions, without metric helper."""
import json
import math
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent
p=pd.read_csv(root/'outputs/weeks7_8/predictions.csv',dtype={'row_id':str,'fold':str})
r=pd.read_csv(root/'outputs/weeks9_10/targeting_by_fold.csv',dtype={'fold':str})
for row in r.itertuples():
    f=p[(p.dataset==row.dataset)&(p.fold==row.fold)&(p.base_model==row.model)&(p.method==row.method)]
    assert not f.row_id.duplicated().any()
    ranked=f.sort_values(['p','row_id'],ascending=[False,True],kind='stable')
    k=math.ceil(len(f)*row.budget)
    hits=ranked.head(k).y.sum()
    assert row.selected==k and row.hits==hits and row.n==len(f)
    assert abs(row.precision-hits/k)<1e-12
    assert abs(row.recall-hits/f.y.sum())<1e-12
    assert abs(row.lift-(hits/k)/f.y.mean())<1e-12
assert len(r)==420
result={'metric_rows_reconciled':len(r),'final_holdouts_read':False,'passed':True}
(root/'outputs/weeks9_10/validation_checks.json').write_text(json.dumps(result,indent=2))
print(result)
