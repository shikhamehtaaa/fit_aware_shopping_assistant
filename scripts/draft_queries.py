import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-3-flash-preview"
client = genai.Client()

cases = pd.read_csv("evals/cases_draft.csv", dtype=str).fillna("")
products = pd.read_parquet("data/products.parquet").set_index("parent_asin")

PROMPT = """Write one realistic shopper request for the clothing item below, as if someone were describing it from memory or from a Pinterest photo, without having seen the listing.
Rules:
- 12 to 25 words, in the shopper's own casual words.
- Do NOT use the brand name, store name, or model numbers.
- Do NOT copy phrases from the title. Describe the look (silhouette, length, sleeves, fabric feel, style, occasion) in your own words.
- {style}
Return only the request text.

TITLE: {title}
FEATURES: {features}"""


def style_for(i, price):
    if i % 3 == 1:
        return "Include a fit preference, such as wanting one that runs true to size or doesn't run small."
    if i % 3 == 2 and price:
        return f"Include a budget, e.g. 'under ${int(price) + 5}'."
    return "Describe only the look, no fit or price."


def draft(row):
    p = products.loc[row.gold_asin]
    price = pd.to_numeric(p.get("price"), errors="coerce")
    price = None if pd.isna(price) else float(price)
    prompt = PROMPT.format(style=style_for(int(row.id), price),
                           title=p["title"], features=str(p["features"])[:600])
    for attempt in range(5):
        try:
            r = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.8))
            return r.text.strip().strip('"')
        except Exception as e:
            print("retry:", repr(e)[:100])
            time.sleep(4 * (attempt + 1))
    return ""


todo = [r for r in cases.itertuples() if not r.query.strip()]
with ThreadPoolExecutor(max_workers=3) as ex:
    results = list(ex.map(draft, todo))

for r, q in zip(todo, results):
    cases.loc[cases.id == r.id, "query"] = q
    cases.loc[cases.id == r.id, "provenance"] = (
        "silver: LLM-drafted from product title/features (title wording avoided), then human-reviewed")

cases.to_csv("evals/cases_draft.csv", index=False)
print(sum(1 for q in results if q), "queries drafted")