import ast, gzip, json, re
import pandas as pd

rows = []
with gzip.open("data_raw/renttherunway_final_data.json.gz", "rt") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except Exception:
            try:
                rows.append(ast.literal_eval(line))
            except Exception:
                pass

df = pd.DataFrame(rows)
df.columns = [c.replace(" ", "_") for c in df.columns]
print(len(df), "rows;", df.columns.tolist())


def height_in(s):
    m = re.match(r"\s*(\d+)'\s*(\d+)", str(s))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def weight_lbs(s):
    m = re.match(r"\s*(\d+)", str(s))
    return int(m.group(1)) if m else None


out = pd.DataFrame({
    "fit": df["fit"],
    "height_in": df["height"].map(height_in),
    "weight_lbs": df["weight"].map(weight_lbs),
    "size": pd.to_numeric(df["size"], errors="coerce"),
    "category": df["category"],
    "bust_size": df.get("bust_size"),
    "body_type": df.get("body_type"),
    "age": pd.to_numeric(df.get("age"), errors="coerce"),
    "rented_for": df.get("rented_for"),
    "item_id": df["item_id"].astype(str),
})
out = out.dropna(subset=["fit", "height_in", "size"])
out["size"] = out["size"].astype(int)
out.to_parquet("data/rtr.parquet")
print(len(out), "rows kept;", out.fit.value_counts().to_dict())