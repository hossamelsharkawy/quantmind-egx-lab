#!/usr/bin/env python3
"""Build a richer QuantMind-compatible EGX feature parquet from frozen Thndr candles.
All engineered factors are strictly backward-looking at each session close.
No QuantMind source modification and no Thndr writes.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=Path('/Users/ahmedalaslani/.local/share/qlib-11-models-20261011-20261007/snapshot/csv')
OUT=ROOT/'db/feature_snapshots/model_features_core.parquet'
MANIFEST=ROOT/'egx_lab/results/quantmind_egx_feature_parquet_v2_manifest.json'


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta=close.diff()
    gain=delta.clip(lower=0)
    loss=(-delta.clip(upper=0))
    # Wilder-style smoothing; all values use current/past observations only.
    ag=gain.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    al=loss.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=ag/al.replace(0,np.nan)
    out=100-(100/(1+rs))
    out=out.where(al.ne(0),100.0)
    return out


def build_one(f: Path) -> pd.DataFrame:
    d=pd.read_csv(f,parse_dates=['date']).sort_values('date').drop_duplicates('date',keep='last')
    d['symbol']=f.stem.upper(); d['trade_date']=pd.to_datetime(d['date']).dt.tz_localize(None)
    for c in ['open','high','low','close','volume','vwap']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    close=d['close']; vol=d['volume']; amount=d['vwap'].fillna(close)*vol
    ret=close.pct_change(fill_method=None)
    d['amount']=amount; d['is_st']=0
    for n in [1,3,5,10,20,60,120]: d[f'mom_ret_{n}d']=close.pct_change(n,fill_method=None)
    for n in [5,10,20,60,120]:
        ma=close.rolling(n,min_periods=n).mean()
        gap=close/ma-1
        d[f'mom_ma_gap_{n}']=gap; d[f'ma_gap_{n}']=gap
    ema12=close.ewm(span=12,adjust=False,min_periods=12).mean()
    ema26=close.ewm(span=26,adjust=False,min_periods=26).mean()
    dif=ema12-ema26; dea=dif.ewm(span=9,adjust=False,min_periods=9).mean(); hist=dif-dea
    for prefix in ['mom_','']:
        d[f'{prefix}macd_dif']=dif; d[f'{prefix}macd_dea']=dea; d[f'{prefix}macd_hist']=hist
    for n in [6,14]:
        x=rsi(close,n); d[f'mom_rsi_{n}']=x; d[f'rsi_{n}']=x
    for n in [5,10,20,60]: d[f'vol_std_{n}']=ret.rolling(n,min_periods=n).std(ddof=0)
    prev=close.shift(1); tr=pd.concat([(d.high-d.low).abs(),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    for n in [14,20]: d[f'vol_atr_{n}']=tr.rolling(n,min_periods=n).mean()/close
    d['liq_volume']=vol; d['liq_amount']=amount
    for n in [5,10,20]:
        vma=vol.rolling(n,min_periods=n).mean(); ama=amount.rolling(n,min_periods=n).mean()
        d[f'liq_volume_ma_{n}']=vma; d[f'liq_amount_ma_{n}']=ama
        if n in [5,20]:
            d[f'liq_volume_ratio_{n}']=vol/vma.replace(0,np.nan)
            d[f'liq_amount_ratio_{n}']=amount/ama.replace(0,np.nan)
    d['volume_ma_3']=vol.rolling(3,min_periods=3).mean(); d['amount_ma_5']=amount.rolling(5,min_periods=5).mean()
    keep=['trade_date','symbol','open','high','low','close','volume','amount','is_st'] + [c for c in d.columns if c.startswith(('mom_','vol_','liq_','ma_gap_','rsi_','macd_'))] + ['volume_ma_3','amount_ma_5']
    return d[list(dict.fromkeys(keep))]

frames=[build_one(f) for f in sorted(SRC.glob('*.csv'))]
frame=pd.concat(frames,ignore_index=True).sort_values(['trade_date','symbol']).reset_index(drop=True)
assert not frame.duplicated(['trade_date','symbol']).any()
assert frame.symbol.nunique()==266 and str(frame.trade_date.max().date())=='2026-10-07'
for c in frame.columns:
    if c not in {'trade_date','symbol'}: frame[c]=pd.to_numeric(frame[c],errors='coerce').astype('float32')
OUT.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(OUT,index=False,compression='zstd')
sha=hashlib.sha256(OUT.read_bytes()).hexdigest()
feature_cols=[c for c in frame.columns if c not in {'trade_date','symbol','open','high','low','close','volume','amount','is_st'}]
manifest={'source':'frozen Thndr daily snapshot','rows':len(frame),'symbols':frame.symbol.nunique(),'first_date':str(frame.trade_date.min().date()),'last_date':str(frame.trade_date.max().date()),'sha256':sha,'output':str(OUT),'feature_count':len(feature_cols),'features':feature_cols,'causality':'all engineered features use session t and earlier only','missing_not_fabricated':['fun_turnover_1'],'mode':'research-only; source read-only'}
MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False,indent=2))
