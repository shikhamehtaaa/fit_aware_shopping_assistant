from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Fit Shopping Agent</h1><p>Coming soon.</p>"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest):
    # swap this for your Project 1 response shape
    return {"answer": f"stub: {req.message}", "tool_calls": []}