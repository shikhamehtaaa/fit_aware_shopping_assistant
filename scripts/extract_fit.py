import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from google import genai
from google.genai import types
import random

MODEL = "gemini-3-flash-preview"
CACHE = "data_raw/fit_cache.jsonl"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None

client = genai.Client()
products = pd.read_parquet("data/products.parquet")
reviews = pd.read_parquet("data/reviews.parquet")

try:
    CONFIG = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
except Exception:
    CONFIG = types.GenerateContentConfig(
        response_mime_type="application/json", temperature=0
    )

PROMPT = """You are extracting sizing/fit information from clothing reviews.
For each review below, return an object with:
- review_id (copy exactly)
- fit: "small" (runs small / too tight), "large" (runs large / too loose), "true" (true to size / fits well), or "unclear"
- areas: list of any of [bust, waist, hips, shoulders, sleeves, length, thighs] the reviewer says are a problem
- reviewer_height: the reviewer's height as stated (e.g. "5'4"), else null
- size_purchased: the size they bought (e.g. "M", "8"), else null
Only use what the review explicitly says. Use "unclear" rather than guessing.
Return a JSON list of these objects.

Reviews:
{reviews}"""


def extract(asin):
    rs = reviews[reviews.parent_asin == asin]
    block = "\n".join(f"[{r.review_id}] {(r.text or '')[:500]}" for r in rs.itertuples())
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=PROMPT.format(reviews=block),
                config=CONFIG,
            )
            result = json.loads(resp.text)
            if isinstance(result, list):
                return asin, result
        except Exception as e:
            print("error:", repr(e)[:300])
            time.sleep(min(60, 3 * 2 ** attempt) + random.random() * 3)
    print("failed", asin)
    return asin, None


done = set()
if os.path.exists(CACHE):
    with open(CACHE) as f:
        done = {json.loads(l)["asin"] for l in f}

todo = [a for a in products.parent_asin if a not in done]
if LIMIT:
    todo = todo[:LIMIT]
print(len(todo), "products to process")

with ThreadPoolExecutor(max_workers=6) as ex, open(CACHE, "a") as out:
    for i, (asin, result) in enumerate(ex.map(extract, todo)):
        if result is not None:
            out.write(json.dumps({"asin": asin, "items": result}) + "\n")
            out.flush()
        if i % 25 == 0:
            print("done", i)

rows = []
with open(CACHE) as f:
    for line in f:
        d = json.loads(line)
        for it in d["items"]:
            rows.append({
                "review_id": it.get("review_id"), "parent_asin": d["asin"],
                "fit": it.get("fit"), "areas": ",".join(it.get("areas") or []),
                "reviewer_height": it.get("reviewer_height"),
                "size_purchased": it.get("size_purchased"),
            })
if not rows:
    sys.exit("No successful extractions, check the errors above.")
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
print(len(ext), "extractions;", len(summ), "products summarized")