import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

MODEL = "gemini-3-flash-preview"
MAX_TURNS = 6
DATA = Path(__file__).parent / "data"

# ---------- data ----------
products = pd.read_parquet(DATA / "products.parquet")
reviews = pd.read_parquet(DATA / "reviews.parquet")
ext = pd.read_parquet(DATA / "fit_extractions.parquet")
summ = pd.read_parquet(DATA / "fit_summary.parquet")

df = products.merge(summ, on="parent_asin", how="inner").reset_index(drop=True)
df["price_num"] = pd.to_numeric(df["price"], errors="coerce")


def tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


docs = (
    df["title"].fillna("") + " " + df["store"].fillna("") + " "
    + df["features"].fillna("") + " " + df["description"].fillna("").str[:500]
)
bm25 = BM25Okapi([tokenize(d) for d in docs])
by_asin = {r.parent_asin: i for i, r in enumerate(df.itertuples())}


def card(i):
    r = df.iloc[i]
    return {
        "asin": r.parent_asin,
        "title": r.title,
        "store": r.store,
        "price": None if pd.isna(r.price_num) else float(r.price_num),
        "rating": None if pd.isna(r.average_rating) else float(r.average_rating),
        "image_url": r.image_url,
        "fit": {
            "small": int(r.small), "true": int(r.true), "large": int(r.large),
            "n_fit_reviews": int(r.n_fit_reviews),
        },
    }


# ---------- tools ----------
def search_products(query, max_price=None, min_fit_reviews=3, k=5):
    scores = bm25.get_scores(tokenize(query))
    mask = (df["n_fit_reviews"] >= float(min_fit_reviews)).to_numpy()
    if max_price is not None:
        mask &= (df["price_num"] <= float(max_price)).fillna(False).to_numpy()
    scores = np.where(mask, scores, -1.0)
    top = [int(i) for i in np.argsort(-scores)[: int(k)] if scores[i] > 0]
    if not top:
        return {"results": [], "note": "No matching products in this dataset."}
    return {"results": [card(i) for i in top]}


def get_fit_evidence(asin):
    e = ext[ext.parent_asin == asin]
    if e.empty or asin not in by_asin:
        return {"error": "No fit data for this product."}
    counts = {k: int(v) for k, v in e.fit.value_counts().items()}
    areas = (
        e.areas.fillna("").str.split(",").explode().str.strip()
        .replace("", np.nan).dropna().value_counts()
    )
    body = e[e.reviewer_height.notna() | e.size_purchased.notna()].head(6)
    m = e.merge(reviews[["review_id", "text"]], on="review_id")
    snippets = []
    for label in ["small", "large", "true"]:
        for r in m[m.fit == label].head(2).itertuples():
            snippets.append({"review_id": r.review_id, "fit": label,
                             "text": (r.text or "")[:220]})
    return {
        "asin": asin,
        "title": df.iloc[by_asin[asin]].title,
        "fit_counts": counts,
        "n_reviews_analyzed": int(len(e)),
        "problem_areas": {k: int(v) for k, v in areas.items()},
        "reviewer_details": [
            {"review_id": r.review_id, "fit": r.fit,
             "height": r.reviewer_height if pd.notna(r.reviewer_height) else None,
             "size_bought": r.size_purchased if pd.notna(r.size_purchased) else None}
            for r in body.itertuples()
        ],
        "snippets": snippets,
    }


TOOL_FNS = {"search_products": search_products, "get_fit_evidence": get_fit_evidence}

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_products",
        description=("Keyword search over clothing products that have fit data. "
                     "Query should be concrete garment terms (e.g. 'cropped cardigan "
                     "knit button'), not full sentences. Returns up to k products "
                     "with counts of reviews saying small/true/large."),
        parameters={"type": "OBJECT", "properties": {
            "query": {"type": "STRING"},
            "max_price": {"type": "NUMBER", "description": "Optional USD cap"},
            "min_fit_reviews": {"type": "INTEGER", "description": "Min fit-mentioning reviews, default 3"},
            "k": {"type": "INTEGER", "description": "Number of results, default 5"},
        }, "required": ["query"]},
    ),
    types.FunctionDeclaration(
        name="get_fit_evidence",
        description=("Detailed fit evidence for one product: review counts by fit "
                     "label, problem areas, reviewer heights/sizes where stated, and "
                     "short review snippets with review IDs for citing."),
        parameters={"type": "OBJECT", "properties": {
            "asin": {"type": "STRING"},
        }, "required": ["asin"]},
    ),
])

SYSTEM = """You are a fit-aware clothing shopping assistant. You search a dataset of about {n} Amazon clothing products and use fit evidence mined from customer reviews.

How to work:
1. Turn the user's description into concrete search queries and call search_products (try 2 different queries if the first is weak).
2. For the 2-3 best candidates, call get_fit_evidence before recommending anything.
3. Recommend at most 3 products. Cite products by ASIN in square brackets like [B0ABC12345] and review evidence by review ID.
4. Always report sample sizes (e.g. "9 of 14 fit-mentioning reviews say it runs small"). If fewer than 5 reviews mention fit, say the evidence is thin.
5. If the user gives height or measurements, compare them with reviewer heights/sizes in the evidence, and say when the data has no reviewer similar to them.
6. Be honest about what this data cannot tell you: current stock, today's price, fabric stretch, or anything not in the reviews. If nothing matches, say so instead of stretching.
Keep answers short and practical."""

client = genai.Client()  # Vertex via GOOGLE_GENAI_USE_VERTEXAI env vars


def clean(o):
    return o.item() if hasattr(o, "item") else str(o)


def run_agent(message, history):
    contents = []
    for h in history[-6:]:
        role = "model" if h.get("role") == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=h.get("content", ""))]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM.format(n=len(df)),
        tools=[TOOLS],
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    tool_calls = []
    for _ in range(MAX_TURNS):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        calls = resp.function_calls
        if not calls:
            return resp.text or "", tool_calls
        contents.append(resp.candidates[0].content)
        parts = []
        for c in calls:
            args = dict(c.args or {})
            try:
                result = TOOL_FNS[c.name](**args)
            except Exception as ex:
                result = {"error": str(ex)}
            result = json.loads(json.dumps(result, default=clean))
            tool_calls.append({
                "agent": "main", "tool": c.name, "args": args,
                "result_preview": json.dumps(result)[:600],
            })
            parts.append(types.Part.from_function_response(name=c.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))

    contents.append(types.Content(role="user", parts=[types.Part(
        text="Out of turns. Give your best final answer now using what you found.")]))
    resp = client.models.generate_content(
        model=MODEL, contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM.format(n=len(df)), temperature=0.2),
    )
    return resp.text or "", tool_calls


# ---------- API ----------
app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/health")
def health():
    return {"ok": True, "products": len(df)}


@app.post("/chat")
def chat(req: ChatRequest):
    answer, tool_calls = run_agent(req.message, req.history)
    asins = list(dict.fromkeys(re.findall(r"\b(B[0-9A-Z]{9})\b", answer)))
    cards = [card(by_asin[a]) for a in asins if a in by_asin]
    return {"answer": answer, "tool_calls": tool_calls, "products": cards}


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "static" / "index.html").read_text()
