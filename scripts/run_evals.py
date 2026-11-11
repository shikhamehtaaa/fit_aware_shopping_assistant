import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from google.genai import types
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app

LABEL = sys.argv[1] if len(sys.argv) > 1 else "baseline"
RUN_AGENT = "--agent" in sys.argv

cases = pd.read_csv("evals/cases_draft.csv", dtype=str).fillna("")
cases = cases[cases["query"].str.strip() != ""].reset_index(drop=True)
n = len(cases)
print(f"{n} cases | margin of error ~ +/-{1 / n ** 0.5:.2f}")

# Scorer 1: deterministic retrieval check (no LLM). Is the gold product in the top 5?
hits = []
for r in cases.itertuples():
    res = app.search_products(r.query, min_fit_reviews=0, k=5)
    hits.append(r.gold_asin in [p["asin"] for p in res["results"]])
cases["retrieval_hit"] = hits
print(f"[{LABEL}] retrieval recall@5: {sum(hits) / n:.3f}")

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
    print(f"[{LABEL}] agent found gold: {cases.agent_found_gold.mean():.3f}")
    print(f"[{LABEL}] judge faithfulness: {scored.mean():.3f} (n={len(scored)})")

cases.to_csv(f"evals/results_{LABEL}.csv", index=False)