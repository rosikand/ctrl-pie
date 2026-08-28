from fastapi import FastAPI

app = FastAPI(
    title="ctrl-π API",
    description="Local control plane for the ctrl-π robot-learning console.",
    version="0.1.0",
)


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "mock"}

