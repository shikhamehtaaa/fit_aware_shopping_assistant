# Fit Notes: Evidence-Grounded Clothing Recommendations

## 1. Project Overview

Fit Notes is an AI-powered clothing recommendation assistant designed to help shoppers find products that are more likely to fit them based on customer review data.

General-purpose shopping assistants may recommend products based on descriptions, product features, or broad customer sentiment. Fit Notes goes a step further by retrieving fit-related information from customer reviews, including whether reviewers found an item true to size, too small, or too large.

The application combines product search with review-derived fit evidence to generate recommendations. Users can ask for clothing recommendations in natural language, provide preferences such as budget or fit requirements, and inspect the evidence behind the assistant's suggestions.

**Target audience:** Online clothing shoppers who want recommendations that account for fit, sizing, and the experiences of previous customers.

**Project question:**

Can customer-review-derived fit information help an AI shopping assistant make more reliable and personalized clothing recommendations than a general-purpose language model?

## 2. Dataset

### Dataset description

Fit Notes uses four Parquet files:

| File | Description |
|---|---|
| `products.parquet` | Product information used for searching and displaying recommendations. |
| `reviews.parquet` | Customer reviews associated with products. |
| `fit_extractions.parquet` | Extracted fit-related information from customer reviews. |
| `fit_summary.parquet` | Aggregated fit information for individual products. |

The application combines `products.parquet` and `fit_summary.parquet` using an inner join on `parent_asin`.

The review and fit-related data are used to provide evidence about how customers experienced the fit of individual products.

### Dataset source and provenance

**Dataset source:**

**Original dataset URL or reference:**

**How the dataset was collected:**

**Date or time period covered by the dataset:**

**Number of products:**

**Number of reviews:**

**Number of fit extractions:**

**Other relevant dataset details:**

### Data directory structure

The application expects the following files inside a `data/` directory:

```text
project/
├── app.py
├── data/
│   ├── products.parquet
│   ├── reviews.parquet
│   ├── fit_extractions.parquet
│   └── fit_summary.parquet
├── pyproject.toml
├── uv.lock
└── README.md
```

## 3. Application Architecture

### Product retrieval

Fit Notes uses BM25, a ranking algorithm for text retrieval, to identify products relevant to a user's query.

The searchable product text includes:
- Product title
- Store
- Product features
- The first 500 characters of the product description

The application retrieves relevant products and can filter results using:
- Maximum price
- Minimum number of fit reviews

The default search returns up to five products.

### Agent tools

The Gemini model can call two tools to retrieve information from the dataset.

**`search_products`**

Searches for products relevant to a user's request. It supports an optional maximum price and minimum fit-review count.

**`get_fit_evidence`**

Retrieves fit-related evidence for a specific product. The returned information includes:
- Counts of reviews by fit label
- Common fit problem areas
- Reviewer height and purchased-size details, when available
- Short review excerpts
- Review IDs that can be used to trace the evidence

### Agent workflow

1. The user submits a clothing recommendation request through the frontend.
2. The Gemini model interprets the request and determines which dataset tools are needed.
3. The model searches for relevant products using `search_products`.
4. The model retrieves fit evidence for promising candidates using `get_fit_evidence`.
5. The model combines product information and review-derived fit evidence to produce recommendations.
6. The frontend displays the response, recommended products, and tool-call information.

The application limits the conversation history to the most recent six turns and allows up to six tool-call iterations during a response.

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

- 
- 
- 

### What the application provides

Fit Notes provides product recommendations supported by information retrieved from its product and fit-review datasets. Recommendations are based on the available data and should not be interpreted as guarantees that a product will fit a particular shopper.

## 5. Assignment Path: Evals + Simple RAG

**Selected assignment path:**

### Research questions

1. Can customer reviews be used to identify whether a clothing item generally fits true to size, runs small, or runs large?
2. Can reviewer height and purchased-size information provide more useful fit guidance?
3. Do recommendations grounded in retrieved product and fit-review evidence differ from recommendations produced by a general-purpose language model without access to the dataset?

### Evaluation dataset

**Location of the evaluation questions and labels:**

**Number of labeled evaluation samples:**

**How the evaluation questions were created:**

**How expected answers were labeled (gold/silver):**

**How the source of each evaluation sample was recorded:**

**Confirmation that evaluation samples were not included in the agent's few-shot prompts:**

### Evaluation metrics and scorers

The assignment requires at least two scorer types.

**Scorer 1 — Type and description:**

**Scorer 2 — Type and description:**

**Additional scorers, if applicable:**

**How scorer results are combined or interpreted:**

### Evaluation command

```bash

```

### Baseline evaluation

**Baseline configuration:**

**Baseline evaluation results:**

**Baseline output examples:**

### Iterative improvement / hill-climbing log

The assignment requires a baseline and at least two subsequent changes, with evaluation results recorded for each iteration.

| Version | Change made | Evaluation results | Observations |
|---|---|---|---|
| Baseline |  |  |  |
| Iteration 1 |  |  |  |
| Iteration 2 |  |  |  |

### Margin of error

The assignment specifies a margin of error of approximately:

\[
\pm \frac{1}{\sqrt{n}}
\]

where \(n\) is the number of labeled evaluation samples.

**Number of evaluation samples used:**

**Calculated margin of error:**

**Interpretation:**

## 6. Comparison: Agent vs. General-Purpose Model

To evaluate the value of retrieving product and fit-review evidence, the agent should be compared against a baseline Gemini response that does not have access to the dataset tools or the Fit Notes system prompt.

The same user queries should be used for both configurations.

### Comparison prompts

**Prompt 1:**

**Prompt 2:**

**Prompt 3:**

### Agent outputs

**Prompt 1 — Agent response:**

**Prompt 2 — Agent response:**

**Prompt 3 — Agent response:**

### Baseline outputs

**Prompt 1 — Baseline response:**

**Prompt 2 — Baseline response:**

**Prompt 3 — Baseline response:**

### Comparison and findings

**Key differences between the agent and baseline:**

**Examples of recommendations supported by retrieved evidence:**

**Cases where the agent could not find sufficient evidence:**

**Overall findings:**
