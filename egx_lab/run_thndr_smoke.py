"""Offline QuantMind template gate smoke test on read-only Thndr CSV research snapshot.
Never imports/runs QuantMind broker, backend, secrets, Docker or account services.
Results are exploratory and not proof of point-in-time-valid live trading.
"""
import ast, json, os
from pathlib import Path
import numpy as np
import pandas as pd

FORK=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ["THNDR_RESEARCH_ROOT"]).expanduser()
COST=.002
def config(name):
    code=ast.parse((FORK/"strategy_templates"/(name+".py")).read_text())
    return next(ast.literal_eval(node.value)["kwargs"] for node in code.body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and
        t.id=="STRATEGY_CONFIG" for t in node.targets))

def history(symbol):
    f=ROOT/"snapshot"/"csv"/(symbol+".csv")
    z=pd.read_csv(f,usecols=["date","close","volume"],parse_dates=["date"])
    z=z.rename(columns={"date":"datetime"}).sort_values("datetime")
    z["instrument"]=symbol
    c=z.close.astype(float); delta=c.diff()
    up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    z["rsi"]=100-100/(1+up/down.replace(0,np.nan))
    z["gap20"]=100*(c/c.rolling(20).mean()-1)
    z["gap5"]=100*(c/c.rolling(5).mean()-1)
    z["ma60"]=c.rolling(60).mean()
    z["ma60lag10"]=z.ma60.shift(10)
    z["value5"]=(c*z.volume).rolling(5).mean()
    return z[["datetime","instrument","close","rsi","gap20","gap5",
              "ma60","ma60lag10","value5"]]

def evaluate(z,days,k=5):
    selected=z.loc[z.groupby("datetime").score.rank(ascending=False,method="first")<=k]
    daily=pd.Series(0.0,index=days)
    r=selected.groupby("datetime").net_executable.mean()
    daily.loc[r.index]=r.values
    nav=(1+daily).cumprod()
    return {"trades":int(len(selected)),"active_days":int(selected.datetime.nunique()),
            "winners":int((selected.net_executable>0).sum()),
            "win_rate_pct":round(float((selected.net_executable>0).mean())*100,2)
             if len(selected) else None,
            "net_pct":round(float(nav.iloc[-1]-1)*100,2)}

def main():
    s=pd.read_csv(ROOT/"results"/"quality-gated-11models-20260826-holdout-pretest.csv")
    q=s.loc[s.quality_technical_pass]
    assert len(q)==33
    selected=q.nlargest(22,"median_20d_value_egp").ticker.to_list()
    loc=ROOT/"models"
    a=pd.read_parquet(loc/"mlp_next_open_close_v1"/"test-predictions.parquet",
                      columns=["datetime","instrument","score","net_executable"])
    b=pd.read_parquet(loc/"lightgbm_next_open_close_v1"/"test-predictions.parquet",
                      columns=["datetime","instrument","score"])
    z=a.rename(columns={"score":"mlp"}).merge(
       b.rename(columns={"score":"lgbm"}),on=["datetime","instrument"],validate="one_to_one")
    z.datetime=pd.to_datetime(z.datetime)
    z=z.loc[z.instrument.isin(selected)]
    assert len(z)>=500 and z.datetime.nunique()==26
    hist=pd.concat([history(symbol) for symbol in selected],ignore_index=True)
    z=z.merge(hist,on=["datetime","instrument"],validate="one_to_one")
    t18=config("as18_trend_gate")
    t22=config("as22_pullback_in_uptrend")
    assert t18["trend_ma"]==60 and t18["trend_slope_days"]==10
    liquidity=z.value5>=2_000_000 # EGP-specific adaptation, NOT vendor f_amount units.
    trend=liquidity&(z.close>z.ma60)&(z.ma60>z.ma60lag10)
    pull=liquidity&z.gap20.between(t22["f_ma_gap_20_min"],t22["f_ma_gap_20_max"])&(
        z.rsi<=t22["f_rsi_14_max"])&(z.gap5>=t22["f_ma_gap_5_min"])
    days=sorted(z.datetime.unique())
    rows=[]
    for model in ("mlp","lgbm"):
        for name,mask in [("baseline",pd.Series(True,index=z.index)),
                          ("QuantMind_as18_trend_gate",trend),
                          ("QuantMind_as22_pullback",pull)]:
            p=z.loc[mask,["datetime","instrument",model,"net_executable"]].rename(columns={model:"score"})
            rows.append({"model":model,"template":name,**evaluate(p,days)})
    report={"status":"OFFLINE_RESEARCH_ONLY","data":"Thndr frozen 2026-10-07 snapshot",
            "selection_asof":"2026-08-26","test_window":"2026-09-01..2026-10-06",
            "universe":selected,"stocks":len(selected),"rows":len(z),"test_sessions":len(days),
            "candidates":{"as18":int(trend.sum()),"as22":int(pull.sum())},
            "cost_bps_each_side":20,"slippage_included":False,
            "quantmind_templates":["as18_trend_gate","as22_pullback_in_uptrend"],
            "scope":"Template gates evaluated, not full QuantMind Redis TopK engine",
            "warning":"PIT publication timestamps unverified; EGX T+0 eligibility unverified; no execution authority",
            "results":rows}
    outfile=FORK/"egx_lab"/"results"/"offline_smoke_20261011.json"
    outfile.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("QUANTMIND_EGX_SMOKE",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
