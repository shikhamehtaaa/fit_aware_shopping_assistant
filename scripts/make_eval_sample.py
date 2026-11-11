import os
import pandas as pd

products = pd.read_parquet("data/products.parquet")
summ = pd.read_parquet("data/fit_summary.parquet")
df = products.merge(summ, on="parent_asin")
df = df[df.n_fit_reviews >= 5]
df = df.sample(min(70, len(df)), random_state=7)

out = pd.DataFrame({
    "id": range(1, len(df) + 1),
    "gold_asin": df.parent_asin.values,
    "title": df.title.values,
    "image_url": df.image_url.values,
    "price": df.price.values,
    "query": "", "author": "", "provenance": "",
})
os.makedirs("evals", exist_ok=True)
out.to_csv("evals/cases_draft.csv", index=False)
print(len(out), "rows written")