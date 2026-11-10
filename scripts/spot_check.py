import pandas as pd

e = pd.read_parquet("data/fit_extractions.parquet")
r = pd.read_parquet("data/reviews.parquet")

print(e.fit.value_counts(), "\n")
first = e.parent_asin.unique()
print(len(r[r.parent_asin.isin(first)]), "reviews sent;", len(e), "extracted\n")

m = e.merge(r[["review_id", "text"]], on="review_id")
for row in m.sample(6, random_state=1).itertuples():
    print(row.fit, "|", row.areas, "|", row.reviewer_height, "|", row.size_purchased)
    print(row.text[:300], "\n")