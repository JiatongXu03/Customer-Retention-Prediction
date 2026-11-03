"""Generate development calibration and drift figures from validated tables."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/retention-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json


def build(root):
    dest = root/'outputs/weeks7_8'
    assert json.loads((dest/'validation_checks.json').read_text())['passed']
    metrics = pd.read_csv(dest/'fold_metrics.csv')
    retail = metrics.loc[metrics.dataset.eq('retail')]
    fig, axes = plt.subplots(1,2, figsize=(12,4), layout='constrained')
    for (base,method), block in retail.groupby(['base_model','method']):
        label = base.replace(' fixed candidate 0','')+' / '+method
        axes[0].plot(block.fold,block.brier, marker='o',label=label)
        axes[1].plot(block.fold,block.retention_error_pp,marker='o',label=label)
    axes[0].set_ylabel('Brier score (lower is better)')
    axes[1].set_ylabel('Retention overprediction (percentage points)')
    axes[1].axhline(0,color='gray',linewidth=.7)
    for ax in axes:
        ax.tick_params(axis='x',rotation=25)
    axes[1].legend(fontsize=7)
    fig.savefig(dest/'retail_calibration.png',dpi=160)
    plt.close(fig)
    drift = pd.read_csv(dest/'retail_feature_drift.csv')
    table = drift.loc[drift.reference.eq('first_origin')].pivot(index='feature',columns='cutoff',values='psi')
    fig, ax = plt.subplots(figsize=(9,4),layout='constrained')
    im = ax.imshow(table.to_numpy(),aspect='auto',cmap='YlOrRd')
    ax.set_xticks(range(len(table.columns)),table.columns,rotation=20)
    ax.set_yticks(range(len(table)),table.index)
    ax.set_title('Retail feature PSI relative to first development origin')
    fig.colorbar(im,ax=ax,label='PSI (descriptive)')
    fig.savefig(dest/'retail_feature_drift.png',dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    build(Path(__file__).resolve().parent)
