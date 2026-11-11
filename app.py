import json
import re
from pathlib import Path
import time
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
df = df[~df["title"].str.contains(r"grill|smoker|vent|burner|cookware|filter", case=False, regex=True, na=False)].reset_index(drop=True)
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
def _search(query, max_price=None, min_fit_reviews=3, k=5):
    scores = bm25.get_scores(tokenize(query))
    mask = (df["n_fit_reviews"] >= float(min_fit_reviews)).to_numpy().copy()
    if max_price is not None:
        mask = mask & (df["price_num"] <= float(max_price)).to_numpy()
    scores = np.where(mask, scores, -1.0)
    top = [int(i) for i in np.argsort(-scores)[: int(k)] if scores[i] > 0]
    if not top:
        return {"results": [], "note": "No matching products in this dataset."}
    return {"results": [card(i) for i in top]}

def search_products(query, max_price=None, min_fit_reviews=3, k=5):
    kw, mp = llm_rewrite(query)
    if max_price is None:
        max_price = mp
    return _search(kw or query, max_price, min_fit_reviews, k)

STOP = set("""a an the and or of for to in on with that this it is are be but my me want
looking need something like very really so just usually probably maybe would could can
im ive id dont doesnt wont""".split())
FIT_WORDS = {"runs", "run", "true", "size", "sizes", "fit", "fits", "small", "large",
             "between", "usually", "swallow"}


def prep_query(text):
    """Deterministic cleanup: pull out a price cap, drop filler and fit words."""
    max_price = None
    m = re.search(r"(?:under|below|less than|max|up to)\s*\$?\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        max_price = float(m.group(1))
        text = text.replace(m.group(0), " ")
    toks = [t for t in tokenize(text)
            if t not in STOP and t not in FIT_WORDS and len(t) > 1]
    return " ".join(toks), max_price


REWRITE_PROMPT = """Convert a shopper's clothing request into a keyword search query for a catalog of Amazon clothing listings (titles, features, descriptions).
Return JSON only: {"keywords": "...", "max_price": number or null}
- keywords: 6 to 14 plain words covering garment type, silhouette, length, sleeves, neckline, fabric, pattern, color, style or occasion. Add 3 to 5 synonyms a seller would likely use in a listing (for example 'breezy' -> 'lightweight flowy loose'). No sizes, no fit complaints, no brand names, no price words.
- max_price: the dollar cap if the shopper states a budget, else null.
Request: __Q__"""


def llm_rewrite(text):
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=REWRITE_PROMPT.replace("__Q__", text),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0),
            )
            d = json.loads(r.text)
            mp = d.get("max_price")
            return str(d.get("keywords", "")), (float(mp) if mp is not None else None)
        except Exception:
            time.sleep(3 * (attempt + 1))
    return prep_query(text)  # fall back to the deterministic version


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

rtr = pd.read_parquet(DATA / "rtr.parquet")


def fit_by_body(height_inches, weight_lbs=None, size=None, category=None):
    d = rtr[(rtr.height_in - float(height_inches)).abs() <= 2]
    if weight_lbs is not None:
        d = d[(d.weight_lbs - float(weight_lbs)).abs() <= 15]
    if category:
        d = d[d.category.str.contains(str(category), case=False, na=False)]
    if len(d) < 20:
        return {"error": "Too few similar reviewers in the Rent the Runway data.",
                "n_similar": int(len(d))}

    def rates(x):
        v = x.fit.value_counts(normalize=True)
        return {k: round(float(v.get(k, 0)), 3) for k in ["small", "fit", "large"]}

    if size is not None:
        x = d[d["size"] == int(size)]
        if len(x) < 10:
            return {"error": "Too few similar reviewers who rented that size.",
                    "n_similar": int(len(d)), "n_that_size": int(len(x))}
        return {"n": int(len(x)), "size": int(size), "rates": rates(x)}
    by_size = []
    for s, x in d.groupby("size"):
        if len(x) >= 15:
            by_size.append({"size": int(s), "n": int(len(x)), "rates": rates(x)})
    by_size.sort(key=lambda r: -r["rates"]["fit"])
    return {"n_similar_reviewers": int(len(d)), "sizes": by_size[:5]}

TOOL_FNS = {"search_products": search_products, "get_fit_evidence": get_fit_evidence, "fit_by_body": fit_by_body}

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
        types.FunctionDeclaration(
        name="fit_by_body",
        description=("Rent the Runway data (192k rentals with reviewer measurements): for "
                     "reviewers within about 2 inches of the given height (and 15 lb of the "
                     "given weight), what share said small / fit / large, overall or for one "
                     "numeric US size. This calibrates sizing for a body type; it is NOT "
                     "specific to any Amazon product."),
        parameters={"type": "OBJECT", "properties": {
            "height_inches": {"type": "NUMBER", "description": "Height in inches, e.g. 5'4 = 64"},
            "weight_lbs": {"type": "NUMBER"},
            "size": {"type": "INTEGER", "description": "Numeric US size rented, optional"},
            "category": {"type": "STRING", "description": "e.g. dress, gown, jacket, optional"},
        }, "required": ["height_inches"]},
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
Keep answers short and practical.
7. If the user gives height (ideally with weight or a size), call fit_by_body. It reports how reviewers of similar height and weight fared in Rent the Runway's numeric sizes. It is a body-based calibration, not evidence about any specific product. Report the numeric sizes exactly as returned, never convert them to S/M/L, state the sample size n, and keep it separate from the product evidence.
8. Format: plain text only. No markdown (no #, **, *, or links) and no URLs. One short paragraph per product that starts with the product title and its ASIN in brackets. Cite at most two review IDs per product, in brackets at the end of a sentence. Put the body calibration in its own short paragraph at the end."""

client = genai.Client()  # Vertex via GOOGLE_GENAI_USE_VERTEXAI env vars


def clean(o):
    return o.item() if hasattr(o, "item") else str(o)

def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [scrub(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and o != o:
        return None
    return o


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
            result = scrub(json.loads(json.dumps(result, default=clean)))
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
