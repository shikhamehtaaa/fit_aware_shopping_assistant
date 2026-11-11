# Fit Notes: Evidence-Grounded Clothing Recommendations

**Live demo:** https://fit-aware-shopping-assistant-git-375188753147.europe-west1.run.app
**Path:** Evals + Simple RAG
**Team:** Shikha Mehta (sjm2291), Simryn Parikh (sap2254)

## 1. Project Overview

Fit Notes is an AI-powered clothing recommendation assistant designed to help shoppers find products that are more likely to fit them based on customer review data.

General-purpose shopping assistants may recommend products based on descriptions, product features, or broad customer sentiment. Fit Notes goes a step further by retrieving fit-related information from customer reviews, including whether reviewers found an item true to size, too small, or too large.

The application combines product search with review-derived fit evidence to generate recommendations. Users can ask for clothing recommendations in natural language, provide preferences such as budget or fit requirements, and inspect the evidence behind the assistant's suggestions.

**Target audience:** Online clothing shoppers who want recommendations that account for fit, sizing, and the experiences of previous customers.

**Project question:** Can customer-review-derived fit information help an AI shopping assistant make more reliable and personalized clothing recommendations than a general-purpose language model?

## 2. Dataset

### Dataset description

Fit Notes uses two data sources, stored as five Parquet files:

| File | Description |
|---|---|
| `products.parquet` | Product information used for searching and displaying recommendations. |
| `reviews.parquet` | Customer reviews (those that mention fit) associated with products. |
| `fit_extractions.parquet` | Structured fit information extracted from each review by Gemini: fit label (small / true / large / unclear), problem areas, reviewer height, and size purchased. |
| `fit_summary.parquet` | Aggregated fit label counts for individual products. |
| `rtr.parquet` | Source 2, Rent the Runway fit feedback: per-rental fit verdict (small / fit / large) with the reviewer's height, weight, numeric size rented, and category. Used by the `fit_by_body` tool. |

The application combines `products.parquet` and `fit_summary.parquet` using an inner join on `parent_asin`, so only products that have extracted fit data are searchable.

### Dataset source and provenance

**Dataset sources:**

1. **Amazon Reviews 2023** (McAuley Lab, UC San Diego), `Amazon_Fashion` category: product metadata and customer reviews. Used for product search and product-specific fit evidence.
2. **Rent the Runway clothing fit data** (McAuley Lab, UC San Diego; Misra, Wan, and McAuley, "Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces", RecSys 2018): about 192k rental transactions, each with a fit verdict and the reviewer's body measurements. Used for body-based size calibration.

**Original dataset URL or reference:** Amazon: https://amazon-reviews-2023.github.io [FILL IN: add the citation and license line from the dataset page]. Rent the Runway: https://cseweb.ucsd.edu/~jmcauley/datasets.html (file: https://mcauleylab.ucsd.edu/public_datasets/data/renttherunway/renttherunway_final_data.json.gz).

**How the dataset was collected:**

1. The Amazon_Fashion metadata and review files were downloaded from the dataset site (`data_raw/`, not committed).
2. `scripts/build_data.py` kept products whose title mentions a garment type (dress, top, blouse, shirt, tee, sweater, cardigan, jeans, pants, skirt), that have an image and at least 20 ratings (26,288 candidates), then randomly sampled 9,000 (seed 0).
3. For each sampled product it kept up to 30 reviews that mention fit (matched with a regular expression), and kept only products with at least 5 such reviews (1,739 products, 17,950 reviews).
4. `scripts/extract_fit.py` sent each product's reviews to Gemini (`gemini-3-flash-preview` on Vertex AI) to extract structured fit labels. Extraction was stopped once enough products were processed because of API rate limits, so the app uses the subset that finished. `scripts/build_fit_tables.py` turns the extraction cache into the two parquet files above.
5. Rent the Runway: `renttherunway_final_data.json.gz` was downloaded from the McAuley lab site, and `scripts/build_rtr.py` parsed height (to inches), weight (to pounds), and size, dropped rows missing fit, height, or size, and wrote `data/rtr.parquet`. Review text was dropped to keep the file small.

**Date or time period covered by the dataset:** [FILL IN: review date range from the dataset card; the 2023 release covers reviews through 2023]

**Number of products (in the app):** [FILL IN: see command in the "Rebuilding the data" section]

**Number of reviews:** [FILL IN]

**Number of fit extractions:** [FILL IN]

**Rent the Runway rows kept:** [FILL IN: printed by `scripts/build_rtr.py`]

**Other relevant dataset details:** The fit labels are LLM-extracted, not human-labeled, and the model was told to answer "unclear" rather than guess. Reviewer height and size are only filled in when the review states them. A few non-clothing items (for example grill parts) matched the title filter and are in the corpus.

### Data directory structure

The application expects the following files inside a `data/` directory:

```text
project/
├── app.py
├── static/
│   └── index.html
├── data/
│   ├── products.parquet
│   ├── reviews.parquet
│   ├── fit_extractions.parquet
│   ├── fit_summary.parquet
│   └── rtr.parquet
├── scripts/
├── evals/
├── pyproject.toml
├── uv.lock
└── README.md
```

### Rebuilding the data

```bash
# 1. download the Amazon_Fashion meta and review .jsonl.gz files into data_raw/
uv run python scripts/build_data.py        # sample products and fit-mentioning reviews
uv run python scripts/extract_fit.py       # Gemini fit extraction (needs Vertex AI access)
uv run python scripts/build_fit_tables.py  # build fit_extractions / fit_summary parquet files
# 2nd source: download renttherunway_final_data.json.gz into data_raw/ (link above)
uv run python scripts/build_rtr.py         # Rent the Runway -> data/rtr.parquet
```

## 3. Application Architecture

### Product retrieval

Fit Notes uses BM25, a ranking algorithm for text retrieval, to identify products relevant to a user's query. Before searching, `search_products` rewrites the shopper's request with Gemini into 6 to 14 plain keywords, adds likely seller-language synonyms (for example "breezy" becomes "lightweight flowy loose"), and extracts a budget if one is stated. If the rewrite call fails, it falls back to rule-based cleanup (dropping filler and fit words, parsing a price cap).

The searchable product text includes:
- Product title
- Store
- Product features
- The first 500 characters of the product description

The application can filter results using:
- Maximum price
- Minimum number of fit reviews

The default search returns up to five products.

### Agent tools

The Gemini model can call three tools to retrieve information from the datasets.

**`search_products`**

Searches for products relevant to a user's request (with the rewrite step above). It supports an optional maximum price and minimum fit-review count.

**`get_fit_evidence`**

Retrieves fit-related evidence for a specific product. The returned information includes:
- Counts of reviews by fit label
- Common fit problem areas
- Reviewer height and purchased-size details, when available
- Short review excerpts
- Review IDs that can be used to trace the evidence

**`fit_by_body`**

Uses the Rent the Runway data. Given a height (and optionally weight, a numeric size, or a category), it finds reviewers within about 2 inches of that height (and 15 lb of that weight) and returns the share who said small, fit, or large, either for one size or for the sizes with the highest fit rate. This is a body-based calibration for Rent the Runway's numeric sizes. It is not evidence about any specific Amazon product, and the agent is instructed to keep it separate from the product evidence and report the sample size.

### Agent workflow

1. The user submits a clothing recommendation request through the frontend.
2. The Gemini model interprets the request and determines which dataset tools are needed.
3. The model searches for relevant products using `search_products`.
4. The model retrieves fit evidence for promising candidates using `get_fit_evidence`.
5. If the user gives a height (and ideally weight or size), the model calls `fit_by_body` for a body-based sizing calibration, kept separate from the product evidence.
6. The model combines product information and review-derived fit evidence to produce recommendations.
7. The frontend displays the response, recommended products, and tool-call information.

The `/chat` endpoint returns `answer`, `tool_calls` (each with an `agent` field), and `products`. The application limits the conversation history to the most recent six turns and allows up to six tool-call iterations per response.

### Evidence and limitations

The assistant is instructed to:
- Search using concrete product terms.
- Try another search query if the initial results are weak.
- Inspect multiple product candidates before making recommendations.
- Recommend no more than three products.
- Reference product ASINs and review IDs when presenting evidence.
- Report the number of fit reviews supporting a recommendation.
- Warn users when fewer than five fit reviews are available.
- Compare reviewer height and purchased size when that information is available.
- Avoid making unsupported claims about live inventory, current prices, or fabric stretch.

## 4. Frontend and User Instructions

### How to use Fit Notes

1. **Describe what you are looking for.** Enter a natural-language request describing the clothing item, desired fit, budget, or other preferences.
2. **Review the recommendations.** The assistant searches the product dataset and presents relevant options.
3. **Inspect the evidence.** Review the fit information and tool-call details to understand what supports each recommendation.

Users can also select one of the example prompts displayed in the frontend to try the application.

### Example prompts

The frontend includes the following example prompts:

- A flowy summer dress. I'm 5'4" and between a medium and large.
- High-waisted jeans that don't run small, under $40.
- A soft knit cardigan that fits petite frames.

### What the application provides

Fit Notes provides product recommendations supported by information retrieved from its product and fit-review datasets. Recommendations are based on the available data and should not be interpreted as guarantees that a product will fit a particular shopper.

## 5. Assignment Path: Evals + Simple RAG

**Selected assignment path:** Evals + Simple RAG (`"path": "evals"` in `submission.json`)

### Research questions

1. Can customer reviews be used to identify whether a clothing item generally fits true to size, runs small, or runs large?
2. Can reviewer height and purchased-size information provide more useful fit guidance?
3. Do recommendations grounded in retrieved product and fit-review evidence differ from recommendations produced by a general-purpose language model without access to the dataset?

### Evaluation dataset

**Location of the evaluation questions and labels:** `evals/cases_draft.csv` (columns: `id`, `gold_asin`, `title`, `image_url`, `price`, `query`, `author`, `provenance`)

**Number of labeled evaluation samples:** 70

**How the evaluation questions were created:** 70 products were randomly sampled from the app's corpus (seed 7, at least 5 fit reviews each). Gemini drafted one shopper-style request per product from its title and features, with instructions to avoid brand names and not reuse title wording. About a third include a fit preference and a third include a budget. [FILL IN: names] then reviewed every query, fixing or removing vague ones.

**How expected answers were labeled (gold/silver):** Silver. The expected answer is the product the query was written from (`gold_asin`). The query text is LLM-drafted and human-reviewed.

**How the source of each evaluation sample was recorded:** In the `provenance` and `author` columns of each row.

**Confirmation that evaluation samples were not included in the agent's few-shot prompts:** The agent's system prompt contains no few-shot examples, so none of the evaluation queries appear in any prompt.

### Evaluation metrics and scorers

**Scorer 1: deterministic retrieval check.** For each query, the rewritten search must return the gold product. Reported as recall@5, recall@20, and mean reciprocal rank (MRR).

**Scorer 2: LLM judge with a rubric (`--agent` mode).** The full agent answers each query, and a Gemini judge checks whether every fit claim in the answer (review counts, "runs small", reviewer heights and sizes) is supported by the evidence for the cited products. Score 1 if supported, 0 if not. The same run also reports how often the gold product appears in the agent's final answer.

**How scorer results are combined or interpreted:** Scorers are reported separately. Retrieval metrics drive the hill-climbing log; the judge score measures whether the answers are faithful to the evidence.

### Evaluation command

```bash
uv run python scripts/run_evals.py baseline            # raw BM25 on the full request
uv run python scripts/run_evals.py stopwords           # rule-based query cleanup
uv run python scripts/run_evals.py rewrite             # Gemini keyword rewrite
uv run python scripts/run_evals.py rewrite --agent     # adds end-to-end agent run and judge
```

(Requires Vertex AI access: `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION=global`.)

### Baseline evaluation

**Baseline configuration:** BM25 over title, store, features, and description, using the full unedited shopper request as the query (top 20, no fit-review minimum).

**Baseline evaluation results:** recall@5 = 0.086, recall@20 = 0.143, MRR = 0.057 (n = 70).

**Baseline output examples:** Typical misses were queries that described a product with different words than the listing, for example "scarlet knee-length flared cocktail piece" for a floral lace cap-sleeve dress, and "sheer ankle-length outer layer" for a chiffon swimsuit cover-up.

### Iterative improvement / hill-climbing log

| Version | Change made | Evaluation results | Observations |
|---|---|---|---|
| Baseline | Raw BM25 on the full request | recall@5 0.086, recall@20 0.143, MRR 0.057 | Queries include filler and fit words, and shoppers use different vocabulary than sellers. |
| Iteration 1 | Strip filler and fit words, parse the price cap | recall@5 0.129, recall@20 0.214, MRR 0.075 | Gain of +0.043 in recall@5 is smaller than the margin of error, so it is not distinguishable from noise. |
| Iteration 2 | Gemini rewrites the request into keywords with seller-language synonyms and a budget | recall@5 0.671, recall@20 0.800, MRR 0.509 | Gain of about +0.54 in recall@5 over Iteration 1 is far larger than the margin of error. Fixing the vocabulary gap was the main bottleneck. |

**End-to-end (agent) results** (`rewrite --agent`, n = 70): the gold product appears in the agent's final answer in 74.3% of cases (52 of 70), and judge faithfulness is 0.771 (54 of 70 answers had every fit claim supported by the cited evidence). The end-to-end hit rate is higher than retrieval-only recall@5 (0.671), most likely because the agent can run several searches and recommend up to three products.

### Margin of error

The assignment specifies a margin of error of approximately ±1/√n, where n is the number of labeled evaluation samples.

**Number of evaluation samples used:** 70

**Calculated margin of error:** ±1/√70 ≈ ±0.12

**Interpretation:** A change only counts as a real improvement if it moves a score by more than about 0.12. Iteration 1 does not clear that bar; Iteration 2 does by a wide margin. Gemini rewrites are not deterministic: repeated runs of the rewrite mode gave recall@5 between about 0.57 and 0.69, so the reported numbers are from the final run. The gain over Iteration 1 exceeds the margin in every run. The judge faithfulness score has the same margin (about ±0.12), so it is reported as a measurement, not as an improvement.

## 6. Comparison: Agent vs. General-Purpose Model

To evaluate the value of retrieving product and fit-review evidence, the agent is compared against a baseline Gemini response that has no access to the dataset tools and no Fit Notes system prompt. The same user queries are used for both. Full side-by-side output is generated by `scripts/sample_queries.py` into `evals/sample_queries.md`.

### Comparison prompts

**Prompt 1:** A flowy summer dress. I'm 5'4" and about 140 lbs, between a medium and large.

**Prompt 2:** High-waisted jeans that don't run small, under $40.

**Prompt 3:** What size should I order in a boxy cropped cardigan, and is it in stock in my size today? (deliberately includes something the data cannot answer)

### Agent outputs

**Prompt 1: Agent response:** Recommended three dresses (KIRA Womens Midi Dress [B07BF6L6WM], CHUNTIANRAN Casual Loose Maxi Dress [B097M4MPHR], MSBASIC Flowy A-line Dress [B0795GRR39]) with counts and review IDs: for example 17 of 20 fit-mentioning reviews call the KIRA dress true to size, and 11 of 19 for CHUNTIANRAN, with 6 saying it runs small. It flagged evidence as thin where it was, and ended with a separate Rent the Runway calibration for 5'4" and 140 lbs (n = 27,842 similar reviewers): 75.3% of size 8 rentals fit, 72% of size 12, and 73.8% of size 14.

*Expected answer and source:* B07BF6L6WM has 17 true / 2 small / 1 large fit labels in `data/fit_summary.parquet`, backed by review IDs in `data/reviews.parquet`. The calibration comes from `data/rtr.parquet`.

**Prompt 2: Agent response:** Returned three products (June Julien Jeggings [B01KC7WVHK], Creabygirls Ripped High Waist Jeans [B01331SHAW], PajamaJeans Bootcut [B01595T8J4]) and noted that the price was not listed for any of them. For the first, 18 of 22 fit-mentioning reviews say true to size and 4 say large, with none saying small, which matches the "don't run small" requirement.

*Expected answer and source:* B01KC7WVHK has 18 true / 4 large fit labels in `data/fit_summary.parquet`. The budget could not be verified because the dataset has no price for these items.

**Prompt 3: Agent response:** Returned three cardigans with fit counts (for example 11 of 13 reviews true to size for DRSKIN [B018TZMTBK], and 3 of 6 for YourStyle [B01E9CKGI0], flagged as thin evidence) and stated that current stock and today's price are not in the dataset. It did not make up availability.

*Expected behavior:* give review-based sizing evidence and say the stock question cannot be answered from the data.

### Baseline outputs

**Prompt 1: Baseline response:** General advice on tiered, wrap, and empire-waist dresses, with a Medium-or-Large rule of thumb that depends on fabric stretch, plus brand suggestions (Quince, Old Navy, Abercrombie). No sources and no counts, and the brand suggestions cannot be checked against this dataset.

**Prompt 2: Baseline response:** A list of five budget denim lines (Old Navy, Target's Universal Thread, Walmart, Amazon Essentials, Signature by Levi Strauss) with price ranges and fit claims from memory, plus generic tips about spandex content. Nothing can be traced to a source, and the prices may be out of date.

**Prompt 3: Baseline response:** Asked for the brand, store, and usual size, gave generic sizing rules for boxy and cropped cardigans, and offered to check availability once given a link, which it cannot actually do.

### Comparison and findings

**Key differences between the agent and baseline:** The agent names specific products and backs each fit claim with review counts and review IDs that can be looked up in `data/reviews.parquet`, and its counts match the evidence it retrieved (see the evidence blocks in `evals/sample_queries.md`). It keeps the Rent the Runway body calibration separate from product evidence and reports sample sizes and thin evidence. The baseline gives general sizing and styling advice and names brands and prices from memory that cannot be checked. In Prompt 3 the baseline also offers to check live stock, which it cannot do, while the agent states that stock is not in its data.

**Examples of recommendations supported by retrieved evidence:** The KIRA midi dress (17 of 20 fit-mentioning reviews true to size, with reviewers at 5'4" cited) for Prompt 1, and the June Julien jeggings (18 of 22 true to size, 4 large, none small) for Prompt 2.

**Cases where the agent could not find sufficient evidence:** Prompt 3 asks about live stock, which is not in the data. In Prompt 2, none of the three products had a listed price, so the "under $40" requirement could not be verified. In Prompt 3 the cardigans returned were not confirmed to be boxy or cropped, and the answer did not say so. Several products had only 6 to 10 fit-mentioning reviews, which the agent flagged as thin.

**Overall findings:** Grounding in review evidence makes the answers more specific and checkable than the baseline, at the cost of a small catalog (a few hundred products), so the agent can only recommend what is in the data, where a general model will always offer generic suggestions. The Rent the Runway calibration was only weakly informative here: fit rates were similar across sizes 8, 12, and 14 (72% to 75%), probably because renters tend to pick sizes that work for them. The LLM judge scored 77% of answers as fully supported (see section 5), so some claims still go beyond the cited evidence.

## 7. Limitations

- The evaluation queries were drafted by an LLM from product titles, and the rewrite step uses the same model family, so the measured gain from rewriting may be flattered. Real shopper phrasing could score lower.
- The corpus has a few hundred products with many near-duplicates (for example, similar dresses), so recall@5 cannot reach 100%.
- Fit labels are LLM-extracted and can be wrong. A few non-clothing items passed the title filter, and one evaluation case's gold product is such an item (a grill part), which counts as a miss.
- Judge faithfulness of 0.77 means roughly a quarter of answers contained at least one fit claim the judge could not verify against the evidence. The judge only sees the evidence for up to three cited products, so it may be stricter than a human reader.
- Prices and ratings come from the dataset snapshot, not live listings, and stock is not available.
- Fit counts reflect past reviewers, who may not match the shopper's body or preferences.
- The Rent the Runway data covers designer rental items in numeric sizes, from people renting for events, so `fit_by_body` calibrates body-to-size patterns only. It says nothing about a specific Amazon product and its sizes do not convert to S/M/L.
- The agent uses a preview Gemini model, which may change or be rate-limited.

## 8. Deployment

Deployed on Google Cloud Run with continuous deployment from this GitHub repository (buildpacks). Gemini is accessed through Vertex AI using the service account, with `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` set as environment variables. The data files are committed to the repo.