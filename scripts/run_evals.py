import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from google.genai import types

import app

MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline"  # baseline | stopwords | rewrite
RUN_AGENT = "--agent" in sys.argv

cases = pd.read_csv("evals/cases_draft.csv", dtype=str).fillna("")
cases = cases[cases["query"].str.strip() != ""].reset_index(drop=True)
n = len(cases)
print(f"{n} cases | margin of error ~ +/-{1 / n ** 0.5:.2f} | mode: {MODE}")

CACHE_PATH = "evals/rewrite_cache.json"
cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
if MODE == "rewrite":
    todo = [q for q in cases["query"] if q not in cache]
    with ThreadPoolExecutor(max_workers=3) as ex:
        for q, out in zip(todo, ex.map(app.llm_rewrite, todo)):
            cache[q] = list(out)
    json.dump(cache, open(CACHE_PATH, "w"))


def get_query(text):
    if MODE == "stopwords":
        return app.prep_query(text)
    if MODE == "rewrite":
        return tuple(cache[text])
    return text, None


# Scorer 1 (deterministic): where does the gold product rank in the top 20?
ranks = []
for r in cases.itertuples():
    q, mp = get_query(r.query)
    res = app.search_products(q, max_price=mp, min_fit_reviews=0, k=20)
    asins = [p["asin"] for p in res["results"]]
    ranks.append(asins.index(r.gold_asin) + 1 if r.gold_asin in asins else None)
cases["rank"] = ranks
r5 = sum(1 for x in ranks if x and x <= 5) / n
r20 = sum(1 for x in ranks if x) / n
mrr = sum(1 / x for x in ranks if x) / n
print(f"[{MODE}] recall@5: {r5:.3f} | recall@20: {r20:.3f} | MRR: {mrr:.3f}")

misses = cases[cases["rank"].isna()].head(4)
for r in misses.itertuples():
    print("  MISS:", r.query[:90], "| gold:", r.title[:60])

# Scorer 2 (rubric judge) + end-to-end check, only with --agent
JUDGE = """You are grading a shopping assistant's answer against evidence.
Score 1 if every fit claim in the answer (review counts, 'runs small', reviewer heights or sizes, sample sizes) is supported by the evidence JSON, or if the answer makes no fit claims and says the evidence is thin or missing. Score 0 if any fit claim contradicts or is absent from the evidence.
Return JSON only: {{"score": 0 or 1, "reason": "one sentence"}}

ANSWER:
{answer}

EVIDENCE:
{evidence}"""


def agent_case(r):
    for attempt in range(4):
        try:
            answer, _ = app.run_agent(r.query, [])
            asins = list(dict.fromkeys(re.findall(r"\b(B[0-9A-Z]{9})\b", answer)))
            ev = {a: app.get_fit_evidence(a) for a in asins[:3]}
            resp = app.client.models.generate_content(
                model=app.MODEL,
                contents=JUDGE.format(answer=answer, evidence=json.dumps(ev, default=str)[:7000]),
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
            )
            return r.gold_asin in asins, json.loads(resp.text).get("score")
        except Exception as e:
            print("retry:", repr(e)[:120])
            time.sleep(5 * (attempt + 1))
    return False, None


if RUN_AGENT:
    with ThreadPoolExecutor(max_workers=3) as ex:
        out = list(ex.map(agent_case, list(cases.itertuples())))
    cases["agent_found_gold"] = [o[0] for o in out]
    cases["judge_score"] = [o[1] for o in out]
    scored = cases.judge_score.dropna()
    print(f"[{MODE}] agent found gold: {cases.agent_found_gold.mean():.3f}")
    print(f"[{MODE}] judge faithfulness: {scored.mean():.3f} (n={len(scored)})")

cases.to_csv(f"evals/results_{MODE}{'_agent' if RUN_AGENT else ''}.csv", index=False)