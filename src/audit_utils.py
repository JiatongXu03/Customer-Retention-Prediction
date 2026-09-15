from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

def project_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / 'config/study_design.json').exists():
            return p
    raise FileNotFoundError('Run from the extracted Retention_Week1 folder or notebooks subfolder.')

def profile(df, dataset):
    records=[]
    for c in df.columns:
        s=df[c]
        blank=s.astype('string').str.strip().eq('').fillna(False)
        rec=dict(dataset=dataset,field=c,dtype=str(s.dtype),rows=len(s),null_count=int(s.isna().sum()),
                 blank_count=int(blank.sum()),missing_or_blank_count=int((s.isna() | blank).sum()),
                 distinct_non_null=int(s.nunique()))
        if pd.api.types.is_numeric_dtype(s):
            rec.update(min=float(s.min()),max=float(s.max()))
        records.append(rec)
    return pd.DataFrame(records)

def load_telecom(root):
    ibm=pd.read_csv(root/'data/raw/ibm_telco.csv')
    iran=pd.read_csv(root/'data/raw/iranian_churn.csv')
    return ibm,iran

def load_retail(root):
    sheets=pd.read_excel(root/'data/raw/online_retail_ii.xlsx',sheet_name=None)
    # Keep source columns for provenance, but exclude them from duplicate detection.
    out=[]
    for name,df in sheets.items():
        df['source_sheet']=name
        out.append(df)
    return pd.concat(out,ignore_index=True)

def clean_retail(raw):
    x=raw.rename(columns={'Invoice':'invoice','StockCode':'stock_code','Description':'description',
                          'Quantity':'quantity','InvoiceDate':'invoice_date','Price':'unit_price',
                          'Customer ID':'customer_id','Country':'country'}).copy()
    x['invoice_date']=pd.to_datetime(x['invoice_date'],errors='coerce')
    base=[c for c in x if c!='source_sheet']
    flags={'missing_customer_id':x.customer_id.isna(),'invalid_date':x.invoice_date.isna(),
           'cancellation_invoice':x.invoice.astype(str).str.upper().str.startswith('C'),
           'nonpositive_quantity':x.quantity.le(0)|x.quantity.isna(),
           'nonpositive_price':x.unit_price.le(0)|x.unit_price.isna(),
           'exact_duplicate_extra':x.duplicated(base)}
    marginal=pd.DataFrame([{'issue':k,'rows':int(v.sum()),'share':float(v.mean())} for k,v in flags.items()])
    remaining=pd.Series(True,index=x.index); flow=[{'step':'raw invoice lines','remaining_rows':len(x),'removed_rows':0}]
    for k,v in flags.items():
        removed=int((remaining&v).sum());remaining &= ~v
        flow.append({'step':k,'remaining_rows':int(remaining.sum()),'removed_rows':removed})
    valid=x.loc[remaining].copy()
    valid['customer_id']=valid.customer_id.astype('int64').astype(str)
    valid['invoice']=valid.invoice.astype(str)
    valid['line_value']=valid.quantity*valid.unit_price
    return valid,marginal,pd.DataFrame(flow)

def telecom_splits(ibm,iran,seed=5920):
    result={}
    for name,df in [('ibm',ibm),('iranian',iran)]:
        y=df.Churn.eq('Yes').astype(int).to_numpy() if name=='ibm' else df.Churn.to_numpy()
        if name=='ibm':
            a,b=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=seed)
            groups=df.customerID.astype(str).to_numpy()
        else:
            # No verified customer ID exists: group identical candidate predictor profiles.
            # Exclude the uncertain Status and formula-derived Customer Value from primary X.
            cols=[c for c in df if c not in ['Churn','Status','Customer Value']]
            groups=pd.util.hash_pandas_object(df[cols],index=False).astype(str).to_numpy()
            a,b=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(df,y,groups))
        split=np.full(len(df),'development',dtype=object);split[b]='reserved_test'
        assert not (set(groups[a]) & set(groups[b]))
        result[name]=pd.DataFrame({'row_index':np.arange(len(df)),'split':split,'group_id':groups})
    return result

def cohort(valid,cutoff,lookback=180,horizon=90):
    cutoff=pd.Timestamp(cutoff);start=cutoff-pd.Timedelta(days=lookback);end=cutoff+pd.Timedelta(days=horizon)
    history=valid.loc[(valid.invoice_date>=start)&(valid.invoice_date<cutoff)].copy()
    future=valid.loc[(valid.invoice_date>=cutoff)&(valid.invoice_date<end)]
    g=history.groupby('customer_id')
    X=g.agg(last_purchase=('invoice_date','max'),frequency_180d=('invoice','nunique'),
            gross_spend_180d=('line_value','sum'),line_count_180d=('invoice','size'))
    X['recency_days']=(cutoff-X.last_purchase).dt.total_seconds()/86400
    label=f'repeat_purchase_{horizon}d'
    X[label]=X.index.isin(future.customer_id).astype(int)
    X[f'churn_proxy_{horizon}d']=1-X[label]
    X['cutoff']=cutoff.isoformat();X['lookback_start']=start.isoformat();X['label_end_exclusive']=end.isoformat()
    assert X.index.is_unique and X.recency_days.gt(0).all() and X.recency_days.le(lookback).all()
    assert set(X[label].unique()).issubset({0,1})
    # last_purchase is audit metadata; future outcomes are never model features.
    return X.reset_index()
