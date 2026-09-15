"""First-stage baselines. No final-test evaluation and no hyperparameter search."""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                            precision_score, recall_score, f1_score, brier_score_loss,
                            log_loss, confusion_matrix)
from sklearn.exceptions import ConvergenceWarning
from audit_utils import load_telecom,cohort

def telecom_development(root):
    ibm,iran=load_telecom(root);out={}
    for name,raw in [('ibm',ibm),('iranian',iran)]:
        assignment=pd.read_csv(root/f'outputs/tables/{name}_split_assignments.csv',dtype={'group_id':str})
        selected=assignment.loc[assignment.split.eq('development')].copy()
        idx=selected.row_index.to_numpy()
        dev=raw.iloc[idx].copy().reset_index(drop=True)
        y=dev.Churn.eq('Yes').astype(int) if name=='ibm' else dev.Churn.astype(int)
        X=dev.drop(columns=['Churn','customerID'] if name=='ibm' else ['Churn','Status','Customer Value']).copy()
        if name=='ibm':
            X['TotalCharges']=pd.to_numeric(X.TotalCharges,errors='coerce')
            cats=[c for c in X if c not in ['tenure','MonthlyCharges','TotalCharges']]
        else:cats=['Complains','Age Group','Tariff Plan']
        for c in cats:X[c]=X[c].astype(str)
        out[name]={'X':X,'y':y.to_numpy(),'groups':selected.group_id.to_numpy(),
                   'row_ids':idx,'raw':dev,'categorical':cats,'reserved_rows':int(assignment.split.eq('reserved_test').sum())}
    return out

def preprocessing(X,categorical,log_numeric=False):
    nums=[c for c in X if c not in categorical]
    steps=[('impute',SimpleImputer(strategy='median',add_indicator=True))]
    if log_numeric:steps.append(('log1p',FunctionTransformer(np.log1p,feature_names_out='one-to-one')))
    steps.append(('scale',StandardScaler()))
    branches=[('numeric',Pipeline(steps),nums)]
    if categorical:
        branches.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
             ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical))
    return ColumnTransformer(branches,remainder='drop',verbose_feature_names_out=True)

def baseline_models(X,categorical,log_numeric=False):
    pp=preprocessing(X,categorical,log_numeric)
    return {'Prior / majority':Pipeline([('prepare',clone(pp)),('model',DummyClassifier(strategy='prior'))]),
            'Logistic Regression':Pipeline([('prepare',clone(pp)),('model',LogisticRegression(C=1.0,solver='lbfgs',max_iter=3000,class_weight=None))])}

def score(y,p,threshold=.5,rankable=True):
    y=np.asarray(y);p=np.asarray(p);pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    k=max(1,int(np.ceil(.1*len(y))))
    # Stable seeded tie-breaking; dummy capacity metrics left undefined because it cannot rank.
    tie=np.random.default_rng(5920).random(len(y));order=np.lexsort((tie,-p))[:k]
    precision_at=float(y[order].mean()) if rankable else np.nan
    return {'n':len(y),'positive_rate':float(y.mean()),'roc_auc':roc_auc_score(y,p),
            'average_precision':average_precision_score(y,p),'accuracy':accuracy_score(y,pred),
            'precision_05':precision_score(y,pred,zero_division=0),'recall_05':recall_score(y,pred,zero_division=0),
            'f1_05':f1_score(y,pred,zero_division=0),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]),
            'precision_top10':precision_at,'recall_top10':float(y[order].sum()/y.sum()) if rankable else np.nan,
            'lift_top10':precision_at/y.mean() if rankable else np.nan,
            'predicted_retention':float(1-p.mean()),'observed_retention':float(1-y.mean()),
            'retention_error_pp':float((y.mean()-p.mean())*100),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def evaluate_one_split(X,y,train,val,categorical,dataset,fold,groups=None,log_numeric=False,row_ids=None):
    if groups is not None:assert not set(groups[train]) & set(groups[val])
    records=[];predictions=[];coefs=[]
    for name,model in baseline_models(X,categorical,log_numeric).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(X.iloc[train],y[train])
        convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
        if convergence:raise RuntimeError('; '.join(convergence))
        p=model.predict_proba(X.iloc[val])[:,1]
        records.append({'dataset':dataset,'fold':str(fold),'model':name,'train_n':len(train),
                        **score(y[val],p,rankable=name=='Logistic Regression')})
        for j,prob in zip(val,p):
            predictions.append({'dataset':dataset,'fold':str(fold),'model':name,'row_id':str(row_ids[j]) if row_ids is not None else str(j),
                                'y':int(y[j]),'p':float(prob),'group_id':str(groups[j]) if groups is not None else ''})
        if name=='Logistic Regression':
            coefs.extend({'dataset':dataset,'fold':str(fold),'feature':f,'coefficient':float(v)}
                         for f,v in zip(model.named_steps['prepare'].get_feature_names_out(),model.named_steps['model'].coef_[0]))
        # Pipeline is intentionally fitted only on this training partition, never reused across folds.
    return records,predictions,coefs

def telecom_baselines(root):
    rows=[];preds=[];coefs=[];fold_ids=[]
    for name,d in telecom_development(root).items():
        splitter=StratifiedKFold(5,shuffle=True,random_state=5920) if name=='ibm' else StratifiedGroupKFold(5,shuffle=True,random_state=5920)
        for i,(train,val) in enumerate(splitter.split(d['X'],d['y'],d['groups'] if name=='iranian' else None),1):
            a,b,c=evaluate_one_split(d['X'],d['y'],train,val,d['categorical'],name,i,groups=d['groups'],row_ids=d['row_ids'])
            rows+=a;preds+=b;coefs+=c
            fold_ids.extend({'dataset':name,'row_index':int(d['row_ids'][j]),'validation_fold':i,'group_id':str(d['groups'][j])} for j in val)
    return pd.DataFrame(rows),pd.DataFrame(preds),pd.DataFrame(coefs),pd.DataFrame(fold_ids)

RETAIL_FEATURES=['recency_days','frequency_180d','gross_spend_180d','unique_products_180d',
                 'frequency_recent90d','frequency_previous90d','observed_purchase_span_days']

def retail_features(valid,cutoff,horizon=90):
    t=pd.Timestamp(cutoff);start=t-pd.Timedelta(days=180)
    out=cohort(valid,t,horizon=horizon).set_index('customer_id')
    hist=valid.loc[(valid.invoice_date>=start)&(valid.invoice_date<t)]
    g=hist.groupby('customer_id')
    out['unique_products_180d']=g.stock_code.nunique()
    out['observed_purchase_span_days']=(g.invoice_date.max()-g.invoice_date.min()).dt.total_seconds()/86400
    recent=hist.loc[hist.invoice_date>=t-pd.Timedelta(days=90)].groupby('customer_id').invoice.nunique()
    prev=hist.loc[hist.invoice_date<t-pd.Timedelta(days=90)].groupby('customer_id').invoice.nunique()
    out['frequency_recent90d']=recent.reindex(out.index,fill_value=0)
    out['frequency_previous90d']=prev.reindex(out.index,fill_value=0)
    assert np.isfinite(out[RETAIL_FEATURES].to_numpy()).all() and out[RETAIL_FEATURES].ge(0).all().all()
    return out.reset_index()

def retail_baselines(root,valid):
    cfg=json.loads((root/'config/study_design.json').read_text())['retail']
    dates=cfg['rolling_development_cutoffs'];reserved=pd.Timestamp(cfg['reserved_test_cutoff'])
    cohorts=[]
    for t in dates:
        assert pd.Timestamp(t)<reserved and pd.Timestamp(t)+pd.Timedelta(days=90)<=reserved
        cohorts.append(retail_features(valid,t))
    rows=[];preds=[];coefs=[];windows=[]
    for i in range(1,len(cohorts)):
        train=pd.concat(cohorts[:i],ignore_index=True);val=cohorts[i]
        assert pd.to_datetime(train.label_end_exclusive).max()<=pd.Timestamp(dates[i])
        all_data=pd.concat([train,val],ignore_index=True)
        X=all_data[RETAIL_FEATURES];y=all_data.churn_proxy_90d.to_numpy()
        ti=np.arange(len(train));vi=np.arange(len(train),len(all_data))
        # Customers may recur over time: use temporal visibility, not customer-disjoint splits.
        a,b,c=evaluate_one_split(X,y,ti,vi,[],'retail',dates[i],log_numeric=True,
                               row_ids=(all_data.customer_id.astype(str)+'@'+all_data.cutoff).to_numpy())
        rows+=a;preds+=b;coefs+=c
        windows.append({'fold':dates[i],'train_origins':','.join(dates[:i]),'train_n':len(train),'train_unique_customers':train.customer_id.nunique(),
                        'validation_n':len(val),'validation_label_end':val.label_end_exclusive.iloc[0],
                        'train_latest_label_end':train.label_end_exclusive.max()})
    return pd.DataFrame(rows),pd.DataFrame(preds),pd.DataFrame(coefs),pd.DataFrame(windows),cohorts

def rate_table(df,group,target):
    out=df.groupby(group,observed=True,dropna=False)[target].agg(n='size',events='sum').reset_index()
    out['rate']=out.events/out.n
    z=1.96;den=1+z*z/out.n;center=(out.rate+z*z/(2*out.n))/den
    half=z*np.sqrt(out.rate*(1-out.rate)/out.n+z*z/(4*out.n**2))/den
    out['wilson_low']=center-half;out['wilson_high']=center+half
    out[group]=out[group].astype(str)
    return out
