from google import genai

PROJECT = "ieor-4570-project-2"
MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
]
LOCATIONS = ["global", "us-central1", "us-east1", "us-east4", "europe-west2"]

for loc in LOCATIONS:
    client = genai.Client(vertexai=True, project=PROJECT, location=loc)
    for m in MODELS:
        try:
            r = client.models.generate_content(model=m, contents="say hi")
            print("WORKS:", m, "|", loc, "|", r.text.strip()[:30])
        except Exception as e:
            print("no:   ", m, "|", loc, "|", str(e)[:60].replace("\n", " "))