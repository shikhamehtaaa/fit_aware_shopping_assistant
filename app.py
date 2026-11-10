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


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fit-Aware Shopping Assistant</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}
h1{margin-bottom:.2rem}.sub{color:#555;margin-top:0}
textarea{width:100%;box-sizing:border-box;padding:.7rem;font-size:1rem;border:1px solid #bbb;border-radius:8px}
button{padding:.6rem 1.1rem;font-size:1rem;border:0;border-radius:8px;background:#222;color:#fff;cursor:pointer}
.chip{background:#eee;color:#222;font-size:.85rem;margin:.2rem .3rem .2rem 0}
#answer{white-space:pre-wrap;margin:1rem 0;line-height:1.5}
.cards{display:flex;gap:.8rem;flex-wrap:wrap}.card{width:215px;border:1px solid #ddd;border-radius:10px;padding:.6rem;font-size:.85rem}
.card img{width:100%;height:210px;object-fit:cover;border-radius:6px}
details{margin-top:1rem;font-size:.85rem}pre{white-space:pre-wrap;background:#f6f6f6;padding:.5rem;border-radius:6px}
</style></head><body>
<h1>Fit-Aware Shopping Assistant</h1>
<p class="sub">Describe a clothing item and your measurements. The agent searches ~N_PRODUCTS Amazon fashion products and checks what thousands of reviews say about how each one actually fits. Built for people who are tired of guessing sizes online. It only knows this dataset: no live stock or prices, and it will say when the reviews don't answer your question.</p>
<textarea id="q" rows="3" placeholder="e.g. A boxy cropped cardigan under $30. I'm 5'3&quot; and tops usually swallow me."></textarea>
<div><button onclick="ask()">Find it</button>
<button class="chip" onclick="fill(this)">flowy summer dress, I'm 5'4" and between a M and L</button>
<button class="chip" onclick="fill(this)">high-waisted jeans that don't run small, under $40</button></div>
<div id="answer"></div><div class="cards" id="cards"></div><div id="trace"></div>
<script>
function fill(b){document.getElementById('q').value=b.textContent}
async function ask(){
 const q=document.getElementById('q').value.trim(); if(!q)return;
 const a=document.getElementById('answer'),c=document.getElementById('cards'),t=document.getElementById('trace');
 a.textContent='Searching and reading reviews...';c.innerHTML='';t.innerHTML='';
 try{
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q})});
  const d=await r.json(); a.textContent=d.answer;
  (d.products||[]).forEach(p=>{const e=document.createElement('div');e.className='card';
   const im=document.createElement('img');im.src=p.image_url;im.referrerPolicy='no-referrer';e.appendChild(im);
   const s=document.createElement('div');s.textContent=p.title+(p.price?' - $'+p.price:'')+' | fit reviews: '+p.fit.small+' small, '+p.fit.true+' true, '+p.fit.large+' large';
   e.appendChild(s);c.appendChild(e)});
  const det=document.createElement('details');const sm=document.createElement('summary');sm.textContent='Show work ('+d.tool_calls.length+' tool calls)';det.appendChild(sm);
  const pre=document.createElement('pre');pre.textContent=JSON.stringify(d.tool_calls,null,2);det.appendChild(pre);t.appendChild(det);
 }catch(e){a.textContent='Something went wrong: '+e}
}
</script></body></html>""".replace("N_PRODUCTS", str(len(df)))


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE