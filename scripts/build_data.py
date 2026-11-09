import gzip, json, re, random, os, urllib.request
import pandas as pd

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
META_URL = f"{BASE}/meta_categories/meta_Amazon_Fashion.jsonl.gz"
REV_URL = f"{BASE}/review_categories/Amazon_Fashion.jsonl.gz"
META_PATH, REV_PATH = "data_raw/meta.jsonl.gz", "data_raw/reviews.jsonl.gz"

TYPE_RE = re.compile(r"\b(dress|top|blouse|shirt|tee|sweater|cardigan|jeans|pants|skirt)\b", re.I)
FIT_RE = re.compile(r"runs? (small|large|big|short|long)|true to size|too (tight|loose|short|long|small|big)|size (up|down)|\bfit(s|ted)?\b", re.I)
N_PRODUCTS, MAX_REVIEWS = 9000, 30
random.seed(0)

def download(url, path):
    if not os.path.exists(path):
        print("downloading", url)
        urllib.request.urlretrieve(url, path)

def first_image(images):
    # handles list-of-dicts or dict-of-lists
    if isinstance(images, list) and images:
        d = images[0]
        return d.get("large") or d.get("hi_res") or d.get("thumb")
    if isinstance(images, dict):
        for k in ("large", "hi_res", "thumb"):
            if images.get(k):
                return images[k][0]
    return None

download(META_URL, META_PATH)
download(REV_URL, REV_PATH)

# 1) products
cands = []
with gzip.open(META_PATH, "rt") as f:
    for line in f:
        m = json.loads(line)
        title = m.get("title") or ""
        img = first_image(m.get("images"))
        if (TYPE_RE.search(title) and img and (m.get("rating_number") or 0) >= 20):
            cands.append({
                "parent_asin": m["parent_asin"], "title": title,
                "store": m.get("store"), "price": m.get("price"),
                "average_rating": m.get("average_rating"),
                "rating_number": m.get("rating_number"),
                "features": " | ".join(m.get("features") or []),
                "description": " ".join(m.get("description") or []),
                "image_url": img,
            })
print("candidate products:", len(cands))
products = random.sample(cands, min(N_PRODUCTS, len(cands)))
keep = {p["parent_asin"] for p in products}

# 2) reviews that mention fit
counts, rows = {}, []
with gzip.open(REV_PATH, "rt") as f:
    for i, line in enumerate(f):
        if i % 1_000_000 == 0:
            print("review lines:", i)
        r = json.loads(line)
        pa = r.get("parent_asin")
        if pa in keep and counts.get(pa, 0) < MAX_REVIEWS and FIT_RE.search(r.get("text") or ""):
            counts[pa] = counts.get(pa, 0) + 1
            rows.append({"review_id": f"{pa}_{i}", "parent_asin": pa,
                         "rating": r.get("rating"), "title": r.get("title"),
                         "text": r.get("text")})

reviews = pd.DataFrame(rows)
# drop products with too few fit-mentioning reviews
good = {k for k, v in counts.items() if v >= 5}
products = pd.DataFrame([p for p in products if p["parent_asin"] in good])
reviews = reviews[reviews.parent_asin.isin(good)]
products.to_parquet("data/products.parquet")
reviews.to_parquet("data/reviews.parquet")
print(len(products), "products,", len(reviews), "reviews")