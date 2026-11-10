
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
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
    history: list[dict] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"ok": True, "products": len(df)}


@app.post("/chat")
def chat(req: ChatRequest):
    answer, tool_calls = run_agent(req.message, req.history)
    asins = list(dict.fromkeys(re.findall(r"\b(B[0-9A-Z]{9})\b", answer)))
    cards = [card(by_asin[a]) for a in asins if a in by_asin]
    return {"answer": answer, "tool_calls": tool_calls, "products": cards}


# ---------- FRONTEND ----------
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#f6f5ef">
<title>Fit Notes — Find your fit</title>

<style>
:root {
  --paper: #f6f5ef;
  --surface: #fffefa;
  --ink: #202820;
  --muted: #73796f;
  --line: #e5e5da;
  --green: #315b45;
  --green-dark: #244633;
  --soft-green: #e9efe8;
  --sand: #eeece2;
  --shadow: 0 12px 34px rgba(37, 49, 38, .07);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.55;
}

button, textarea { font: inherit; }
button { cursor: pointer; }

.page {
  width: min(960px, calc(100% - 40px));
  margin: 0 auto;
  padding: 30px 0 64px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 66px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: .92rem;
  font-weight: 750;
  letter-spacing: -.03em;
}

.brand-mark {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border-radius: 11px;
  background: var(--green);
  color: white;
  font-family: Georgia, serif;
  font-size: 1.2rem;
  font-style: italic;
}

.top-note {
  color: var(--muted);
  font-size: .78rem;
  letter-spacing: .08em;
  text-transform: uppercase;
}

/* Hero section */

.hero {
  max-width: 690px;
  margin: 0 auto 34px;
  text-align: center;
}

.eyebrow {
  color: var(--green);
  font-size: .74rem;
  font-weight: 750;
  letter-spacing: .16em;
  text-transform: uppercase;
}

h1 {
  margin: 13px 0 12px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(2.8rem, 7vw, 4.8rem);
  font-weight: 400;
  letter-spacing: -.065em;
  line-height: .98;
}

h1 em {
  color: var(--green);
  font-weight: 400;
}

.intro {
  max-width: 590px;
  margin: 0 auto;
  color: var(--muted);
  font-size: 1rem;
}

/* Search box */

.search-panel {
  max-width: 760px;
  margin: 0 auto;
  padding: 13px;
  border: 1px solid var(--line);
  border-radius: 19px;
  background: var(--surface);
  box-shadow: var(--shadow);
}

textarea {
  display: block;
  width: 100%;
  min-height: 112px;
  resize: vertical;
  padding: 14px 15px;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--ink);
  font-size: 1rem;
  line-height: 1.6;
}

textarea::placeholder { color: #9a9d93; }

.composer-bottom {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 3px 2px 12px;
}

.hint {
  color: #92968c;
  font-size: .76rem;
}

.primary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 9px;
  min-height: 43px;
  padding: 0 19px;
  border: 0;
  border-radius: 11px;
  background: var(--green);
  color: white;
  font-size: .9rem;
  font-weight: 700;
  transition: background .18s ease, transform .18s ease;
}

.primary:hover {
  background: var(--green-dark);
  transform: translateY(-1px);
}

.primary:disabled {
  opacity: .55;
  cursor: wait;
  transform: none;
}

.arrow { font-size: 1.1rem; }

/* Example prompts */

.examples {
  max-width: 760px;
  margin: 17px auto 0;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.example {
  padding: 8px 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: transparent;
  color: #626b60;
  font-size: .78rem;
  transition: all .18s ease;
}

.example:hover {
  border-color: #a9b9aa;
  background: var(--soft-green);
  color: var(--green-dark);
}

/* How to use section */

.guide {
  max-width: 760px;
  margin: 30px auto 0;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.guide-card {
  padding: 17px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255, 254, 250, .7);
}

.guide-number {
  color: var(--green);
  font-family: Georgia, serif;
  font-size: 1.3rem;
}

.guide-card h3 {
  margin: 7px 0 5px;
  font-size: .86rem;
  font-weight: 750;
}

.guide-card p {
  margin: 0;
  color: var(--muted);
  font-size: .78rem;
  line-height: 1.55;
}

/* Dataset explanation */

.dataset-note {
  max-width: 760px;
  margin: 17px auto 0;
  padding: 15px 17px;
  border-radius: 12px;
  background: var(--soft-green);
  color: #526454;
  font-size: .8rem;
  line-height: 1.6;
}

.dataset-note strong {
  color: var(--green-dark);
}

/* Results and conversation */

.results {
  max-width: 820px;
  margin: 42px auto 0;
}

.section-label {
  margin: 0 0 13px;
  color: var(--muted);
  font-size: .72rem;
  font-weight: 750;
  letter-spacing: .13em;
  text-transform: uppercase;
}

.conversation {
  display: grid;
  gap: 13px;
}

.message {
  max-width: 88%;
  padding: 15px 18px;
  border-radius: 16px;
  white-space: pre-wrap;
  line-height: 1.65;
  font-size: .94rem;
}

.message.user {
  justify-self: end;
  background: var(--green);
  color: white;
  border-bottom-right-radius: 5px;
}

.message.assistant {
  justify-self: start;
  background: var(--surface);
  border: 1px solid var(--line);
  border-bottom-left-radius: 5px;
}

/* Loading state */

.status {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: .88rem;
  padding: 12px 2px;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #d8dfd6;
  border-top-color: var(--green);
  border-radius: 50%;
  animation: spin .8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Product cards */

.product-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 15px;
  margin-top: 25px;
}

.product-card {
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 15px;
  background: var(--surface);
  box-shadow: 0 5px 18px rgba(37, 49, 38, .04);
}

.product-image-wrap {
  height: 225px;
  padding: 10px;
  background: #f0efe8;
}

.product-card img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  mix-blend-mode: multiply;
}

.product-info {
  padding: 13px 14px 15px;
}

.product-title {
  margin: 0 0 7px;
  font-size: .88rem;
  font-weight: 700;
  line-height: 1.4;
}

.product-meta {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  color: var(--muted);
  font-size: .78rem;
}

.fit-line {
  margin-top: 11px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
  color: #5d675c;
  font-size: .75rem;
}

.fit-line strong { color: var(--green-dark); }

/* Tool-call trace */

.trace {
  margin-top: 26px;
  border-top: 1px solid var(--line);
  padding-top: 15px;
}

details summary {
  color: var(--muted);
  cursor: pointer;
  font-size: .8rem;
  font-weight: 650;
}

pre {
  overflow-x: auto;
  margin-top: 12px;
  padding: 15px;
  border-radius: 11px;
  background: #202820;
  color: #e9eee6;
  font-size: .74rem;
  line-height: 1.55;
}

/* Footer */

.footer {
  margin: 58px auto 0;
  max-width: 760px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
  color: #898e83;
  font-size: .76rem;
  line-height: 1.7;
  text-align: center;
}

.footer strong {
  color: #626b60;
  font-weight: 700;
}

/* Mobile layout */

@media (max-width: 600px) {
  .page {
    width: min(100% - 26px, 960px);
    padding-top: 19px;
  }

  .topbar { margin-bottom: 49px; }

  .top-note { font-size: .66rem; }

  .hero { margin-bottom: 25px; }

  .intro { font-size: .91rem; }

  .search-panel { border-radius: 15px; }

  .composer-bottom { align-items: flex-end; }

  .hint { max-width: 150px; }

  .guide { grid-template-columns: 1fr; }

  .message { max-width: 96%; }

  .product-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }

  .product-image-wrap { height: 170px; }
}
</style>
</head>

<body>
<main class="page">

  <!-- Header -->

  <header class="topbar">
    <div class="brand">
      <span class="brand-mark">f.</span>
      <span>fit notes</span>
    </div>
    <div class="top-note">Thoughtful shopping, better fit</div>
  </header>

  <!-- Project purpose and target audience -->

  <section class="hero">
    <div class="eyebrow">Review-powered clothing recommendations</div>

    <h1>
      Find your fit.<br>
      <em>Not just your size.</em>
    </h1>

    <p class="intro">
      Fit Notes is an AI shopping assistant for people who want clothing
      recommendations backed by real customer fit experiences. It explores
      patterns in clothing reviews to help you find products that better
      match your body, usual size, and preferences.
    </p>
  </section>

  <!-- Search input -->

  <form class="search-panel" id="search-form">
    <textarea
      id="q"
      rows="3"
      aria-label="Describe what you're looking for"
      placeholder="Example: I'm 5'4&quot;, usually a medium, and looking for jeans under $40 that don't run small..."
    ></textarea>

    <div class="composer-bottom">
      <span class="hint">
        Include your usual size, height, or budget if helpful.
      </span>

      <button class="primary" id="ask-button" type="submit">
        Find my fit <span class="arrow">↗</span>
      </button>
    </div>
  </form>

  <!-- Example searches -->

  <div class="examples" aria-label="Example searches">
    <button
      class="example"
      type="button"
      data-prompt="A flowy summer dress. I'm 5'4&quot; and between a medium and large."
    >
      A summer dress
    </button>

    <button
      class="example"
      type="button"
      data-prompt="High-waisted jeans that don't run small, under $40."
    >
      Jeans under $40
    </button>

    <button
      class="example"
      type="button"
      data-prompt="A soft knit cardigan that fits petite frames."
    >
      Petite-friendly knits
    </button>
  </div>

  <!-- Instructions for using the agent -->

  <section class="guide" aria-label="How to use Fit Notes">

    <article class="guide-card">
      <div class="guide-number">01</div>
      <h3>Describe what you want</h3>
      <p>
        Enter a clothing item, style, budget, or fit preference.
        Include your height and usual size if you want more personalized
        guidance.
      </p>
    </article>

    <article class="guide-card">
      <div class="guide-number">02</div>
      <h3>Explore the fit evidence</h3>
      <p>
        The agent searches its clothing dataset and examines customer
        reviews for patterns in sizing, fit, and reported problem areas.
      </p>
    </article>

    <article class="guide-card">
      <div class="guide-number">03</div>
      <h3>Compare recommendations</h3>
      <p>
        Review the suggested products, fit-review counts, and supporting
        evidence. You can ask follow-up questions to refine your search.
      </p>
    </article>

  </section>

  <!-- Explanation of the underlying dataset -->

  <div class="dataset-note">
    <strong>What makes Fit Notes different?</strong>
    Recommendations are grounded in a clothing dataset and fit information
    extracted from customer reviews—not just a language model's general
    knowledge. The available evidence varies by product, and reviewer
    height or size information may not always be available.
  </div>

  <!-- Chat results -->

  <section class="results" id="results" aria-live="polite" hidden>
    <p class="section-label">Your fit search</p>

    <div class="conversation" id="conversation"></div>

    <div class="product-grid" id="products"></div>

    <div class="trace" id="trace"></div>
  </section>

  <!-- Limitations -->

  <footer class="footer">
    <strong>Fit Notes</strong> is a review-based clothing recommendation
    research project. Product prices and fit patterns come from the
    available dataset; availability, current prices, and inventory are
    not live-checked. Fit information reflects customer reviews and
    should be treated as guidance, not a guarantee of how an item will fit.
  </footer>

</main>

<script>
const form = document.getElementById('search-form');
const input = document.getElementById('q');
const askButton = document.getElementById('ask-button');
const results = document.getElementById('results');
const conversation = document.getElementById('conversation');
const productGrid = document.getElementById('products');
const trace = document.getElementById('trace');
const history = [];

document.querySelectorAll('[data-prompt]').forEach(button => {
  button.addEventListener('click', () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
});

function addMessage(role, content) {
  const bubble = document.createElement('div');
  bubble.className = 'message ' + role;
  bubble.textContent = content;
  conversation.appendChild(bubble);
  return bubble;
}

function showProducts(products) {
  productGrid.replaceChildren();

  (products || []).forEach(product => {
    const card = document.createElement('article');
    card.className = 'product-card';

    if (product.image_url) {
      const imageWrap = document.createElement('div');
      imageWrap.className = 'product-image-wrap';

      const image = document.createElement('img');
      image.src = product.image_url;
      image.alt = product.title || 'Clothing product';
      image.loading = 'lazy';
      image.referrerPolicy = 'no-referrer';
      image.onerror = () => imageWrap.remove();

      imageWrap.appendChild(image);
      card.appendChild(imageWrap);
    }

    const info = document.createElement('div');
    info.className = 'product-info';

    const title = document.createElement('h3');
    title.className = 'product-title';
    title.textContent = product.title || 'Product';
    info.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'product-meta';

    const price = document.createElement('span');
    price.textContent = product.price != null
      ? '$' + Number(product.price).toFixed(2)
      : 'Price unavailable';

    const rating = document.createElement('span');
    rating.textContent = product.rating != null
      ? '★ ' + Number(product.rating).toFixed(1)
      : '';

    meta.append(price, rating);
    info.appendChild(meta);

    const fit = product.fit || {};
    const fitLine = document.createElement('div');
    fitLine.className = 'fit-line';

    fitLine.innerHTML = '<strong>Fit mentions</strong> · ' +
      Number(fit.small || 0) + ' small · ' +
      Number(fit.true || 0) + ' true · ' +
      Number(fit.large || 0) + ' large';

    info.appendChild(fitLine);

    card.appendChild(info);
    productGrid.appendChild(card);
  });
}

function showToolCalls(calls) {
  trace.replaceChildren();

  if (!calls || !calls.length) return;

  const details = document.createElement('details');
  const summary = document.createElement('summary');

  summary.textContent = 'How this search worked · ' +
    calls.length + ' tool calls';

  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(calls, null, 2);

  details.append(summary, pre);
  trace.appendChild(details);
}

form.addEventListener('submit', async event => {
  event.preventDefault();

  const message = input.value.trim();

  if (!message || askButton.disabled) return;

  results.hidden = false;

  addMessage('user', message);

  const loading = document.createElement('div');
  loading.className = 'status';

  loading.innerHTML =
    '<span class="spinner"></span>' +
    '<span>Looking through products and fit reviews…</span>';

  conversation.appendChild(loading);

  productGrid.replaceChildren();
  trace.replaceChildren();

  askButton.disabled = true;
  askButton.textContent = 'Searching…';

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        history: history.slice(-6)
      })
    });

    if (!response.ok) {
      throw new Error(
        'The server returned an error (' + response.status + ').'
      );
    }

    const data = await response.json();

    loading.remove();

    addMessage(
      'assistant',
      data.answer || 'I could not find a useful answer for that search.'
    );

    history.push({ role: 'user', content: message });
    history.push({
      role: 'assistant',
      content: data.answer || ''
    });

    showProducts(data.products);
    showToolCalls(data.tool_calls);

    input.value = '';

  } catch (error) {
    loading.remove();

    addMessage(
      'assistant',
      'Sorry, something went wrong. ' +
      error.message +
      ' Please try again.'
    );

  } finally {
    askButton.disabled = false;
    askButton.innerHTML =
      'Find my fit <span class="arrow">↗</span>';
  }
});
</script>

</body>
</html>""".replace("N_PRODUCTS", str(len(df)))


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
