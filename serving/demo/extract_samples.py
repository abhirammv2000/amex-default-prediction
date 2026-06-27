"""Pick a few real test customers spanning the risk spectrum and bundle their
raw statements as demo presets (anonymized public Kaggle data)."""
import sys, json, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
from pathlib import Path
sys.path.insert(0, "src")
import config
SP = r"C:/Users/abhir/AppData/Local/Temp/claude/c--Users-abhir-Projects-amex/065b234b-effb-477e-b1f3-153fb37d3c8b/scratchpad"
scores = pd.read_parquet(SP + "/portfolio_scores.parquet")
targets = {"Very low risk":0.02,"Low risk":0.12,"Medium risk":0.35,
           "High risk":0.65,"Very high risk":0.93}
picks, used = {}, set()
for label, t in targets.items():
    cand = scores[~scores.customer_id.isin(used)]
    cid = cand.iloc[(cand.probability_of_default - t).abs().argmin()].customer_id
    picks[label] = cid; used.add(cid)
ids = list(picks.values())
# pushdown filter: read ONLY these customers' statement rows
sub = pq.read_table(config.TEST_PARQUET,
                    filters=[(config.ID_COL, "in", ids)]).to_pandas()
sub.to_parquet("serving/demo/sample_customers.parquet", index=False)
si = scores.set_index("customer_id")
meta = [{"label":l,"customer_ID":c,"pd":float(si.loc[c,"probability_of_default"])}
        for l,c in picks.items()]
Path("serving/demo/sample_meta.json").write_text(json.dumps(meta, indent=2))
print("saved", len(ids), "customers,", len(sub), "statements")
for m in meta: print(f"  {m['label']:16s} pd={m['pd']:.3f}")
