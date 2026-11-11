import os, sys, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app

QUERIES = [
    "A flowy summer dress. I'm 5'4\" and about 140 lbs, between a medium and large.",
    "High-waisted jeans that don't run small, under $40.",
    "What size should I order in a boxy cropped cardigan, and is it in stock in my size today?",
]


def retry(fn):
    for a in range(5):
        try:
            return fn()
        except Exception as e:
            print("retry:", repr(e)[:100])
            time.sleep(5 * (a + 1))
    return None


lines = []
for i, q in enumerate(QUERIES, 1):
    res = retry(lambda: app.run_agent(q, []))
    answer, calls = res if res else ("(failed)", [])
    plain = retry(lambda: app.client.models.generate_content(model=app.MODEL, contents=q).text) or "(failed)"
    asins = list(dict.fromkeys(re.findall(r"\b(B[0-9A-Z]{9})\b", answer)))

    src = []
    for a in asins[:3]:
        ev = app.get_fit_evidence(a)
        if "error" not in ev:
            src.append(f"- {a}: {ev['title'][:80]} | fit_counts={ev['fit_counts']} | "
                       f"n_reviews_analyzed={ev['n_reviews_analyzed']} | problem_areas={ev['problem_areas']}")

    lines += [f"## Query {i}", f"> {q}", "",
              f"### Agent answer ({len(calls)} tool calls)", answer, "",
              "### No-tools model answer", plain, "",
              "### Evidence the agent's answer rests on (data/fit_extractions.parquet, data/fit_summary.parquet)",
              *(src or ["- none"]), "", "---", ""]

os.makedirs("evals", exist_ok=True)
open("evals/sample_queries.md", "w").write("\n".join(lines))
print("wrote evals/sample_queries.md")