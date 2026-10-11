#!/usr/bin/env python3
"""Materialize the frozen read-only Thndr daily snapshot in QuantMind's local parquet contract.

This adapter does not change QuantMind source code and does not write to Thndr.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=Path('/Users/ahmedalaslani/.local/share/qlib-11-models-20261011-20261007/snapshot/csv')
OUT=ROOT/'db/feature_snapshots/model_features_core.parquet'
MANIFEST=ROOT/'egx_lab/results/quantmind_egx_parquet_manifest.json'
rows=[]
for f in sorted(SRC.glob('*.csv')):
    d=pd.read_csv(f,parse_dates=['date'])
    if d.empty: continue
    d=d.sort_values('date').drop_duplicates('date',keep='last')
    d['symbol']=f.stem.upper()
    d['trade_date']=pd.to_datetime(d['date']).dt.tz_localize(None)
    close=pd.to_numeric(d['close'],errors='coerce')
    vol=pd.to_numeric(d['volume'],errors='coerce')
    vwap=pd.to_numeric(d.get('vwap',close),errors='coerce')
    d['mom_ret_1d']=close.pct_change(1,fill_method=None)
    d['mom_ret_5d']=close.pct_change(5,fill_method=None)
    d['mom_ret_20d']=close.pct_change(20,fill_method=None)
    d['liq_volume']=vol
    d['liq_amount']=vwap*vol
    d['amount']=d['liq_amount']
    d['is_st']=0
    keep=['trade_date','symbol','open','high','low','close','volume','amount','is_st',
          'mom_ret_1d','mom_ret_5d','mom_ret_20d','liq_volume','liq_amount']
    rows.append(d[keep])
frame=pd.concat(rows,ignore_index=True)
frame=frame.sort_values(['trade_date','symbol']).reset_index(drop=True)
assert not frame.duplicated(['trade_date','symbol']).any()
assert frame.symbol.nunique() >= 250
assert str(frame.trade_date.max().date()) == '2026-10-07'
for c in frame.columns:
    if c not in {'trade_date','symbol'}:
        frame[c]=pd.to_numeric(frame[c],errors='coerce').astype('float32')
OUT.parent.mkdir(parents=True,exist_ok=True)
MANIFEST.parent.mkdir(parents=True,exist_ok=True)
frame.to_parquet(OUT,index=False,compression='zstd')
h=hashlib.sha256(OUT.read_bytes()).hexdigest()
manifest={
 'source':'Thndr frozen daily snapshot exported from research_candles_daily_v2',
 'source_dir':str(SRC),'output':str(OUT),'sha256':h,'rows':len(frame),
 'symbols':int(frame.symbol.nunique()),'first_date':str(frame.trade_date.min().date()),
 'last_date':str(frame.trade_date.max().date()),'columns':list(frame.columns),
 'known_missing_base_feature':'fun_turnover_1 intentionally omitted rather than fabricated',
 'mode':'research-only read-only source; no broker or database writes'
}
MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False,indent=2))
