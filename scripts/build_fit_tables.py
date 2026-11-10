import json
import pandas as pd

reviews = pd.read_parquet("data/reviews.parquet")
rows = []
with open("data_raw/fit_cache.jsonl") as f:
    for line in f:
        d = json.loads(line)
        for it in d["items"]:
            rows.append({
                "review_id": it.get("review_id"), "parent_asin": d["asin"],
                "fit": it.get("fit"), "areas": ",".join(it.get("areas") or []),
                "reviewer_height": it.get("reviewer_height"),
                "size_purchased": it.get("size_purchased"),
            })
ext = pd.DataFrame(rows)
ext = ext[ext.review_id.isin(set(reviews.review_id))]
ext.to_parquet("data/fit_extractions.parquet")

summ = ext.pivot_table(index="parent_asin", columns="fit", values="review_id",
                       aggfunc="count", fill_value=0).reset_index()
for c in ["small", "true", "large", "unclear"]:
    if c not in summ:
        summ[c] = 0
summ["n_fit_reviews"] = summ[["small", "true", "large"]].sum(axis=1)
summ.to_parquet("data/fit_summary.parquet")
print(len(ext), "extractions;", len(summ), "products")