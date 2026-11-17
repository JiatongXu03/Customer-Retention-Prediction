"""Reproduce weeks 9–10 targeting evidence and the demo's aggregate payload."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.targeting import targeting_metrics
ROOT = Path(__file__).resolve().parent

def run():
    cfg = json.loads((ROOT/'config/weeks9_10_experiment.json').read_text())
    source = ROOT/'outputs/weeks7_8/predictions.csv'
    predictions = pd.read_csv(source, dtype={'row_id': str, 'fold': str})
    rows = []
    for key, frame in predictions.groupby(['dataset','fold','base_model','method']):
        for budget in cfg['budgets']:
            rows.append(dict(zip(['dataset','fold','model','method'], key), budget=budget,
                **targeting_metrics(frame.y,frame.p,frame.row_id,budget)))
    result = pd.DataFrame(rows)
    dest = ROOT/'outputs/weeks9_10'
    dest.mkdir(exist_ok=True, parents=True)
    result.to_csv(dest/'targeting_by_fold.csv',index=False)
    summary = result.groupby(['dataset','model','method','budget']).agg(
        folds=('fold','size'), precision=('precision','mean'), recall=('recall','mean'),
        lift=('lift','mean'), lift_min=('lift','min'), lift_max=('lift','max')).reset_index()
    summary.to_csv(dest/'targeting_summary.csv',index=False)
    # Export only aggregates; no customer-level IDs, outcomes or scores in the website.
    payload = dict(summary=summary.to_dict('records'), folds=result.to_dict('records'),
        provenance=dict(source='Weeks 7–8 saved development validation predictions',
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(), final_tests='Unused',
            averaging='Equal weight per fold; retail customers may recur across time',
            ties=cfg['tie_policy']))
    (ROOT/'sites/retention-lab/dist/evidence.json').write_text(json.dumps(payload,allow_nan=False))
    html_path = ROOT/'sites/retention-lab/dist/index.html'
    if html_path.exists():
        import re
        html = html_path.read_text()
        tag = '<script id="embedded-evidence" type="application/json">'+json.dumps(payload,allow_nan=False)+'</script>'
        html = re.sub(r'<script id="embedded-evidence".*?</script>', lambda _: tag, html, flags=re.S) if 'id="embedded-evidence"' in html else html.replace('<script>\nconst budgets',tag+'<script>\nconst budgets')
        html_path.write_text(html)
    (dest/'run_metadata.json').write_text(json.dumps(dict(config=cfg,source_sha256=payload['provenance']['sha256'],
        rows=len(result), prediction_rows=len(predictions)),indent=2))
    print(summary.query('method == "raw" and budget == 0.1').to_string(index=False))

if __name__ == '__main__': run()
